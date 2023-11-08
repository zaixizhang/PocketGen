import binana
import sys
sys.path.append("..")
import rdkit
import rdkit.Chem as Chem
import copy
import pickle
from tqdm.auto import tqdm
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")


atoms = {}
cnt = 0
index_path = '../data/crossdocked_pocket10/index.pkl'
with open(index_path, 'rb') as f:
    index = pickle.load(f)
for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
    if pocket_fn is None: continue
    try:
        path = '../data/crossdocked_pocket10/'
        ligand_path = os.path.join(path, ligand_fn[:-3] + 'pdbqt')
        pocket_path = os.path.join(path, pocket_fn[:-3] + 'pdbqt')
        ligand, receptor = binana.load_ligand_receptor.from_files(ligand_path, pocket_path)
        hbond_inf = binana.interactions.get_hydrogen_bonds(ligand, receptor)
        atoms[pocket_fn[:-3]+ligand_fn[:-3]] = [int(bond[0][bond[0].find('(', 8)+1:-1]) - 1 for bond in hbond_inf['labels']]
        cnt += 1
        if cnt%10000==0:
            np.save('atoms.npy', atoms)
    except:
        continue

print(cnt)
np.save('atoms.npy', atoms)