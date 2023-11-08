import numpy as np
import pickle
import random
import shutil
import os
from shutil import copyfile
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0)
    args = parser.parse_args()

    pocket_score={}
    with open('../ICML23/data/crossdocked_pocket10/index.pkl', 'rb') as f:
        index = pickle.load(f)

    if os.path.exists('./cabbage-file_maker/match'):
        shutil.rmtree('./cabbage-file_maker/match')
    os.mkdir('./cabbage-file_maker/match')

    for (pocket_fn, ligand_fn, _, rmsd_str) in index[args.id*100:args.id*100+100]:
        if pocket_fn is None: continue
        copyfile('../ICML23/data/crossdocked_pocket10/'+pocket_fn, './cabbage-file_maker/match/'+os.path.split(pocket_fn)[1])

