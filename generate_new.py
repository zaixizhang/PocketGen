import sys
import numpy as np
from rdkit import Chem
import os
import argparse

from tqdm import tqdm
import random
import shutil
from vina import Vina

import torch
import esm
from utils.relax import openmm_relax, relax_sdf
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.evaluation.docking_vina import *
from utils.datasets.pl import PocketLigandPairDataset
from utils.data import torchify_dict
from torch_geometric.transforms import Compose
from utils.datasets import *
from utils.transforms import *
from utils.misc import *
from utils.data import *
from torch.utils.data import DataLoader
from models.PD import Pocket_Design_new
from functools import partial
import subprocess
import pickle
import shutil
import multiprocessing as mp


def convert_pdbqt_to_sdf(pdbqt_file, sdf_file):
    mol = next(pybel.readfile("pdbqt", pdbqt_file))
    mol.removeh()
    mol.write("sdf", sdf_file, overwrite=True)


def calculate_vina(id, pro_path, lig_path, output=True):
    size_factor = 1.2
    buffer = 8.0
    if id is not None:
        pro_path = os.path.join(pro_path, str(id) + '_relaxed.pdb')
        lig_path = os.path.join(lig_path, str(id) + '.sdf')
    #openmm_relax(pro_path)
    #relax_sdf(lig_path)
    mol = Chem.MolFromMolFile(lig_path, sanitize=False)
    pos = mol.GetConformer(0).GetPositions()
    center = np.mean(pos, 0)
    ligand_pdbqt = './tmp/' + str(id) + 'lig.pdbqt'
    protein_pqr = './tmp/' + str(id) + 'pro.pqr'
    protein_pdbqt = './tmp/' + str(id) + 'pro.pdbqt'
    lig = PrepLig(lig_path, 'sdf')
    lig.addH()
    lig.get_pdbqt(ligand_pdbqt)

    prot = PrepProt(pro_path)
    prot.addH(protein_pqr)
    prot.get_pdbqt(protein_pdbqt)

    v = Vina(sf_name='vina', seed=0, verbosity=0)
    v.set_receptor(protein_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    x, y, z = (pos.max(0) - pos.min(0)) * size_factor + buffer
    v.compute_vina_maps(center=center, box_size=[x, y, z])
    energy = v.score()
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])
    energy_minimized = v.optimize()
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    v.dock(exhaustiveness=64, n_poses=30)
    score = v.energies(n_poses=1)[0][0]
    print('Score after docking : %.3f (kcal/mol)' % score)
    if output:
        v.write_poses(pro_path[:-4]+'_docked.pdbqt', n_poses=1, overwrite=True)
        convert_pdbqt_to_sdf(pro_path[:-4]+'_docked.pdbqt', pro_path[:-4]+'_docked.sdf')

    return score


def vina_mp(pro_path, lig_path, number_list):
    pool = mp.Pool(16)
    vina_list = []
    func = partial(calculate_vina, pro_path=pro_path, lig_path=lig_path)
    for vina_score in tqdm(pool.imap_unordered(func, number_list), total=len(number_list)):
        if vina_score != None: vina_list.append(vina_score)
    pool.close()
    print('Vina: ', np.average(vina_list))
    return vina_list


def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, residue_dict=None, seq=None, full_seq_idx=None,
                              r10_idx=None):
    instance = {}

    if protein_dict is not None:
        for key, item in protein_dict.items():
            instance['protein_' + key] = item

    if ligand_dict is not None:
        for key, item in ligand_dict.items():
            instance['ligand_' + key] = item

    if residue_dict is not None:
        for key, item in residue_dict.items():
            instance[key] = item

    if seq is not None:
        instance['seq'] = seq

    if full_seq_idx is not None:
        instance['full_seq_idx'] = full_seq_idx

    if r10_idx is not None:
        instance['r10_idx'] = r10_idx

    return instance


def ith_true_index(tensor, i):
    true_indices = torch.nonzero(tensor).squeeze()
    return true_indices[i].item()


def pdb2data(lig_path, pro_path, seq, full_seq_idx, r10_idx, protein_edit_residue, protein_filename, ligand_filename,
             whole_protein_name):
    with open(pro_path, 'r') as f:
        pdb_block = f.read()
    protein = PDBProtein(pdb_block)
    pocket_dict = protein.to_dict_atom()
    residue_dict = protein.to_dict_residue()
    ligand_dict = parse_sdf_file(lig_path)
    residue_dict['protein_edit_residue'] = protein_edit_residue
    dataset = []
    for _ in range(10):
        full_seq_idx1 = random.sample(list(enumerate(full_seq_idx.tolist())), 1)
        true_index = ith_true_index(protein_edit_residue, full_seq_idx1[0][0])
        residue_dict['protein_edit_residue'] = torch.zeros_like(protein_edit_residue).bool()
        residue_dict['protein_edit_residue'][true_index] = True
        full_seq_idx1 = [id for _, id in full_seq_idx1]
        data = from_protein_ligand_dicts(
            protein_dict=torchify_dict(pocket_dict),
            ligand_dict=torchify_dict(ligand_dict),
            residue_dict=torchify_dict(residue_dict),
            seq=seq,
            full_seq_idx=torch.tensor(full_seq_idx1),
            r10_idx=torch.tensor(r10_idx)
        )
        data['protein_filename'] = protein_filename
        data['ligand_filename'] = ligand_filename
        data['whole_protein_name'] = whole_protein_name
        dataset.append(transform(data))
    return dataset


def input_data(args, index):
    protein_filename = index[0]
    ligand_filename = index[1]
    whole_protein_name = index[2]
    protein_edit_residue = index[3]
    seq = index[4]
    full_seq_idx = index[5]
    r10_idx = index[6]
    lig_path = os.path.join(args.source, ligand_filename)
    pro_path = os.path.join(args.source, whole_protein_name)
    with open(pro_path, 'r') as f:
        pdb_block = f.read()
    protein = PDBProtein(pdb_block)
    pocket_dict = protein.to_dict_atom()
    residue_dict = protein.to_dict_residue()
    ligand_dict = parse_sdf_file(lig_path)
    residue_dict['protein_edit_residue'] = protein_edit_residue

    data = from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict),
        residue_dict=torchify_dict(residue_dict),
        seq=seq,
        full_seq_idx=torch.tensor(full_seq_idx),
        r10_idx=torch.tensor(r10_idx)
    )
    data['protein_filename'] = protein_filename
    data['ligand_filename'] = ligand_filename
    data['whole_protein_name'] = whole_protein_name

    return transform(data)


def name2data(name, args):
    pdb_path = os.path.join(args.target, name, name+'.pdb')
    lig_path = os.path.join(args.target, name, name+'_ligand.sdf')
    pocket_path = os.path.join(args.target, name, name+'_pocket.pdb')
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    protein = PDBProtein(pdb_block)
    seq = ''.join(protein.to_dict_residue()['seq'])
    ligand = parse_sdf_file(lig_path, feat=False)
    r10_idx, r10_residues = protein.query_residues_ligand(ligand, radius=10, selected_residue=None, return_mask=False)
    full_seq_idx, _ = protein.query_residues_ligand(ligand, radius=3.5, selected_residue=r10_residues, return_mask=False)
    assert len(r10_idx) == len(r10_residues)

    pdb_block_pocket = protein.residues_to_pdb_block(r10_residues)
    with open(pocket_path, 'w') as f:
            f.write(pdb_block_pocket)

    with open(pocket_path, 'r') as f:
        pdb_block = f.read()
    pocket = PDBProtein(pdb_block)

    pocket_dict = pocket.to_dict_atom()
    residue_dict = pocket.to_dict_residue()
    
    _, residue_dict['protein_edit_residue'] = pocket.query_residues_ligand(ligand)
    assert residue_dict['protein_edit_residue'].sum() > 0 and residue_dict['protein_edit_residue'].sum() == len(full_seq_idx)
    assert len(residue_dict['protein_edit_residue']) == len(r10_idx)
    full_seq_idx.sort()
    r10_idx.sort()
    
            
    data = from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand),
        residue_dict=torchify_dict(residue_dict),
        seq=seq,
        full_seq_idx=torch.tensor(full_seq_idx),
        r10_idx=torch.tensor(r10_idx)
    )
    data['protein_filename'] = pocket_path
    data['ligand_filename'] = lig_path
    data['whole_protein_name'] = pdb_path
    return transform(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_model.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--target', type=str, default='./generate')
    args = parser.parse_args()
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    args.source = config.dataset.path
    seed_all(2089)

    dock_score = []
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
    ])

    # esm
    name = 'esm2_t33_650M_UR50D'
    pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)
    batch_converter = alphabet.get_batch_converter()
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    del pretrained_model

    model = Pocket_Design_new(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    
    print('Loading dataset...')
    names = ['2p16']

    record = [[] for _ in range(len(names))]
    aar_list = [[] for _ in range(len(names))]
    rmsd_list = [[] for _ in range(len(names))]
    attend = []
    for i in tqdm(range(len(names))):
        print(i)
        data = name2data(names[i], args)
        datalist = [data for _ in range(8)]
        protein_filename = data['protein_filename']
        ligand_filename = data['ligand_filename']
        whole_protein_name = data['whole_protein_name']

        print(protein_filename)
        #lig_path = os.path.join(config.dataset.path, ligand_filename)
        #pro_path = os.path.join(config.model.pocket10_path, protein_filename)

        dir_name = os.path.dirname(protein_filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        #original_vina = calculate_vina(None, protein_filename, ligand_filename)
        #record[i].append(original_vina)
        #print('original vina:', original_vina)
        model.generate_id = 0
        model.generate_id1 = 0
        test_loader = DataLoader(datalist, batch_size=4, shuffle=False,
                                    num_workers=config.train.num_workers,
                                    collate_fn=partial(collate_mols_block, batch_converter=batch_converter))
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_loader, desc='Test'):
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(args.device)
                aar, rmsd, attend_logits = model.generate(batch, dir_name)
                attend.append(attend_logits.cpu())
                print('aar: ', aar)
                print('rmsd: ', rmsd)

        score_list = vina_mp(dir_name, dir_name, np.arange(len(datalist)))
        #record[i].extend(score_list)
        
        original_vina = calculate_vina(None, protein_filename, ligand_filename)
        print('original vina:', original_vina)
        with open(os.path.join(args.target, names[i], 'attention.pkl'), 'wb') as file:
            pickle.dump(attend, file)
        #torch.save(record, os.path.join(args.target, 'record.pt'))



