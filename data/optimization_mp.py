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


def calculate_vina(pro_path, lig_path, id, output=False):
    size_factor = 1.
    buffer = 5.
    openmm_relax(pro_path)
    relax_sdf(lig_path)
    mol = Chem.MolFromMolFile(lig_path, sanitize=True)
    pos = mol.GetConformer(0).GetPositions()
    center = np.mean(pos, 0)
    ligand_pdbqt = './data/saved/po1/' + str(id) + 'lig.pdbqt'
    protein_pqr = './data/saved/po1/' + str(id) + 'pro.pqr'
    protein_pdbqt = './data/saved/po1/' + str(id) + 'pro.pdbqt'
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

    return score


def vina_mp(pro_path, lig_path, number_list):
    pool = mp.Pool(16)
    vina_list = []
    for vina_score in tqdm(pool.imap_unordered(partial(calculate_vina, pro_path=pro_path, lig_path=lig_path), number_list), total=len(number_list)):
        if vina_score != None: vina_list.append(vina_score)
    pool.close()
    print('Vina: ', np.average(vina_list))


def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, residue_dict=None, seq=None, full_seq_idx=None, r10_idx=None):
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


def pdb2data(lig_path, pro_path, seq, full_seq_idx, r10_idx, protein_edit_residue, protein_filename, ligand_filename, whole_protein_name):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/test_model.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--source', type=str, default='data/PDBBind_time_split_dataset')
    args = parser.parse_args()
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    optimization_steps = 3

    dock_score = []
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
    ])

    # esm
    name = 'esm1b_t33_650M_UR50S'
    pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)
    batch_converter = alphabet.get_batch_converter()
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    model = Pocket_Design_new(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    with open(os.path.join(args.source, 'index_seq.pkl'), 'rb') as f:
        index = pickle.load(f)

    record = {i: [] for i in range(len(index))}
    for i in tqdm(range(len(index))):
        print(i)
        data_i = input_data(args, index[i])
        protein_filename = index[i][0]
        ligand_filename = index[i][1]
        whole_protein_name = index[i][2]
        protein_edit_residue = torch.tensor(index[i][3])
        seq = index[i][4]
        full_seq_idx = torch.tensor(index[i][5])
        r10_idx = torch.tensor(index[i][6])

        print(protein_filename)
        lig_path = os.path.join(args.source, ligand_filename)
        pro_path = os.path.join(args.source, whole_protein_name)
        try:
            pdb_id = os.path.dirname(protein_filename)
            dir_name = os.path.join('./data/mp_op', pdb_id)
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            min_vina = 0
            original_vina = calculate_vina(pro_path, lig_path)
            record[i].append(original_vina)
            print('original vina:', original_vina)
            for j in range(optimization_steps):
                print('Optimization Steps:', j)
                model.generate_id = 0
                model.generate_id1 = 0
                pro_path = os.path.join(args.source, protein_filename)
                datalist = pdb2data(lig_path, pro_path, seq, full_seq_idx, r10_idx, protein_edit_residue, protein_filename, ligand_filename, whole_protein_name)
                test_loader = DataLoader(datalist, batch_size=2, shuffle=False,
                                         num_workers=config.train.num_workers,
                                         collate_fn=partial(collate_mols_block, batch_converter=batch_converter))
                with torch.no_grad():
                    model.eval()
                    for batch in tqdm(test_loader, desc='Test'):
                        for key in batch:
                            if torch.is_tensor(batch[key]):
                                batch[key] = batch[key].to(args.device)
                        _, _ = model.generate(batch)

                score_list = vina_mp(pro_path, lig_path, np.arange(len(datalist)))
                '''
                for k in range(10):
                    try:
                        score = calculate_vina('./data/mp_op/po1/' + str(k) + '_whole.pdb', './data/mp_op/po1/' + str(k) + '.sdf', output=True)
                    except:
                        score = 0.
                    score_list.append(score)
                '''
                argmin = score_list.index(min(score_list))
                min_vina = min(score_list)
                print('min_vina:', min_vina)
                record[i].append(min_vina)
                pro_path = './data/mp_op/po1/' + str(argmin) + '.pdb'
                lig_path = './data/mp_op/po1/' + str(argmin) + '.sdf'
                whole_pro_path = './data/mp_op/po1/' + str(argmin) + '_whole.pdb'
                docked_path = './data/mp_op/po1/' + str(argmin) + '_whole_docked.pdbqt'
                shutil.copy2(pro_path, os.path.join(dir_name, str(j)+'.pdb'))
                shutil.copy2(whole_pro_path, os.path.join(dir_name, str(j) + '_whole.pdb'))
                shutil.copy2(docked_path, os.path.join(dir_name, str(j) + '_docked.pdbqt'))
                shutil.copy2(lig_path, os.path.join(dir_name, str(j) + '.sdf'))

                with open(pro_path, 'r') as f:
                    pdb_block = f.read()
                protein = PDBProtein(pdb_block)
                new_subset = protein.to_dict_residue()['seq']
                seq_list = list(seq)
                for id, s in enumerate(full_seq_idx):
                    seq_list[s] = new_subset[id]
                seq = ''.join(seq_list)
            torch.save(record, './data/mp_op/po1/record.pt')

        except:
            print('Skip')
