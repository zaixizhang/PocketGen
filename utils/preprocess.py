import sys
import os
sys.path.append("..")
import rdkit
import rdkit.Chem as Chem
import copy
import pickle
from tqdm.auto import tqdm
from protein_ligand import PDBProtein, parse_sdf_file


if __name__ == "__main__":
    NUM_ATOMS = [0, 5, 11, 8, 8, 6, 9, 9, 4, 10, 8, 8, 9, 8, 11, 7, 6, 7, 14, 12, 7]
    cnt = 0
    edit = 0
    num_res = 0
    index_path = '/data/zaixi/Pocket_Design/data/crossdocked_pocket10/index.pkl'
    raw_path = '/data/zaixi/Pocket_Design/data/crossdocked_pocket10'
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index[20000:21000])):
        if pocket_fn is None: continue
        try:
            pdb_data = PDBProtein(os.path.join(raw_path, pocket_fn))
            pocket_dict = pdb_data.to_dict_atom()
            residue_dict = pdb_data.to_dict_residue()
            ligand_dict = parse_sdf_file(os.path.join(raw_path, ligand_fn))
            mask = pdb_data.query_residues_ligand(ligand_dict)
            for k, residue in enumerate(pdb_data.residues):
                if mask[k]:
                    assert len(residue['atoms']) == NUM_ATOMS[pdb_data.AA_NAME_NUMBER[residue['name']]]
            edit += mask.sum()
            num_res+= len(residue_dict['amino_acid'])
            cnt += 1
        except:
            continue

    # number of molecules and vocab
    print('Total number of molecules', cnt)
    print('average residues:', num_res / cnt)
    print('average editable residues:', edit / cnt)
