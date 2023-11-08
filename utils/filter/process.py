import numpy as np
import pickle
import random
import shutil
import os
from shutil import copyfile
import argparse
from vina import Vina
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit import Chem
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()

    pocket_score = {}
    vina_score = []
    index_list = []
    with open('./crossdocked_pocket10/index.pkl', 'rb') as f:
        index = pickle.load(f)

    '''
    if os.path.exists('./vina_score1.npy'):
        vina_score = np.load('vina_score1.npy')
        vina_score = list(vina_score)
    if os.path.exists('./index1.npy'):
        index_list = np.load('index1.npy')
        index_list = list(index_list)'''

    for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index[12637:20000])):
        if pocket_fn is None: continue
        # copyfile('./crossdocked_pocket10/'+pocket_fn, './filter_data/'+os.path.split(pocket_fn)[1])
        # copyfile('./crossdocked_pocket10/' + ligand_fn, './filter_data/' + os.path.split(ligand_fn)[1])
        # copyfile('./crossdocked_pocket10/' + pocket_fn[:-3] + 'pdbqt', './filter_data/' + os.path.split(pocket_fn)[1][:-3]+ 'pdbqt')
        # copyfile('./crossdocked_pocket10/' + ligand_fn[:-3] + 'pdbqt', './filter_data/' + os.path.split(ligand_fn)[1][:-3]+ 'pdbqt')
        try:
            v = Vina(sf_name='vina')
            v.set_receptor('./filter_data/' + os.path.split(pocket_fn)[1][:-3] + 'pdbqt')
            v.set_ligand_from_file('./filter_data/' + os.path.split(ligand_fn)[1][:-3] + 'pdbqt')
            mol = Chem.MolFromMolFile('./filter_data/' + os.path.split(ligand_fn)[1], sanitize=True)
            mol = Chem.AddHs(mol, addCoords=True)
            UFFOptimizeMolecule(mol)
            pos = mol.GetConformer(0).GetPositions()
            center = np.mean(pos, 0)

            v.compute_vina_maps(center=center, box_size=[20, 20, 20])
            #energy = v.score()
            #print('Score before minimization: %.3f (kcal/mol)' % energy[0])
            energy_minimized = v.optimize()
            print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
            if energy_minimized[0] > 10:
                continue
            #v.dock(exhaustiveness=32, n_poses=20)
            vina_score.append(energy_minimized[0])
            index_list.append(i)
            np.save('vina_score2.npy', np.array(vina_score))
            np.save('index2.npy', np.array(index_list))
        except:
            continue
        '''
        if i % 100 == 0:
            np.save('vina_score.npy', np.array(vina_score))
            np.save('index.npy', np.array(index_list))'''
