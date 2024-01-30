import os
import argparse
import multiprocessing as mp
import pickle
import shutil
from functools import partial

from tqdm.auto import tqdm
import openmm.app as app
from openmm.app import PDBFile
from simtk.unit import Quantity
from pdbfixer import PDBFixer
from rdkit import Chem
from rdkit.Chem import rdmolops, rdmolfiles

from utils.protein_ligand import PDBProtein, parse_sdf_file


def load_item(item, path):
    pdb_path = os.path.join(path, os.path.join(item[0][:-4], item[0]))
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    return pdb_block


def remove_mol_H(path):
    mol = Chem.MolFromMolFile(path)
    mol_no_h = rdmolops.RemoveHs(mol)
    writer = rdmolfiles.SDWriter(path)
    writer.write(mol_no_h)
    writer.close()


def removeH(pdb):
    # Load your molecule from a PDB file
    mol = Chem.MolFromPDBFile(pdb, removeHs=False)

    # Remove the hydrogens from the molecule
    mol_no_H = Chem.RemoveHs(mol)

    # Save the modified molecule back to a PDB file
    Chem.MolToPDBFile(mol_no_H, pdb)


def process_name(name):
    try:
        protein_file_name = f"{name}_protein.pdb"

        full_path = os.path.join(args.source, name, protein_file_name)

        removeH(full_path)
    except Exception as e:
        print('Exception occurred for:', name)
        print('Error message:', e)


def collect_result(result, pbar):
    if result is not None:
        index_pocket.append(result)
        pbar.update()


def process_item(item, args):
    try:
        print(item)
        ligand_fn = os.path.join(item[0][:-4], item[1])
        ligand_path = os.path.join(args.source, os.path.join(item[0][:-4], item[1][:-4]+'.sdf'))

        protein_fn = os.path.join(item[0][:-4], item[0])
        pdb_block = load_item(item, args.source)
        protein = PDBProtein(pdb_block)
        seq = ''.join(protein.to_dict_residue()['seq'])
        ligand = parse_sdf_file(ligand_path)
        if len(seq) > 1500:
            return None

        r10_idx, r10_residues = protein.query_residues_ligand(ligand, args.radius, selected_residue=None,
                                                              return_mask=False)
        assert len(r10_idx) == len(r10_residues)

        pdb_block_pocket = protein.residues_to_pdb_block(r10_residues)

        full_seq_idx, _ = protein.query_residues_ligand(ligand, radius=3.5, selected_residue=r10_residues, return_mask=False)

        pocket_fn = os.path.join(item[0][:-4], item[0][:-4] + '_pocket.pdb')
        pocket_dest = os.path.join(args.source, pocket_fn)

        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)
        with open(pocket_dest, 'r') as f:
            pdb_block = f.read()
        pocket = PDBProtein(pdb_block)
        _, protein_edit_residue = pocket.query_residues_ligand(ligand)

        return pocket_fn, ligand_fn, protein_fn, protein_edit_residue, seq, full_seq_idx, r10_idx  # item[0]: original protein filename; item[2]: rmsd.

    except Exception:
        print('Exception occurred.', item)
        return None




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/n/holyscratch01/mzitnik_lab/zaixizhang/BioLiP')
    parser.add_argument('--radius', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    with open(os.path.join(args.source, 'index.pkl'), 'rb') as f:
        index = pickle.load(f)
    '''
    for ind in index:
        result = process_item(ind, args)
    '''

    pool = mp.Pool(args.num_workers)
    index_pocket = []
    for item_pocket in tqdm(pool.imap_unordered(partial(process_item, args=args), index), total=len(index)):
        if item_pocket is not None:
            index_pocket.append(item_pocket)
    pool.close()

    index_path = os.path.join(args.source, 'index_seq.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(index_pocket, f)

    print('Done. %d protein-ligand pairs in total.' % len(index_pocket))

