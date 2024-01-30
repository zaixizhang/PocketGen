import os
import argparse
import random
import torch
from tqdm.auto import tqdm

from torch.utils.data import Subset
from utils.datasets.pl import PocketLigandPairDataset


def get_chain_name(fn):
    return os.path.basename(fn)[:6]


def get_pdb_name(fn):
    return os.path.basename(fn)[:4]


def get_unique_pockets(dataset, raw_id, used_pdb, num_pockets):
    # only save first encountered id for unseen pdbs
    unique_id = []
    pdb_visited = set()
    for idx in tqdm(raw_id, 'Filter'):
        pdb_name = get_pdb_name(dataset[idx].ligand_filename)
        if pdb_name not in used_pdb and pdb_name not in pdb_visited:
            unique_id.append(idx)
            pdb_visited.add(pdb_name)

    print('Number of Pairs: %d' % len(unique_id))
    print('Number of PDBs:  %d' % len(pdb_visited))

    random.Random(args.seed).shuffle(unique_id)
    unique_id = unique_id[:num_pockets]
    print('Number of selected: %d' % len(unique_id))
    return unique_id, pdb_visited.union(used_pdb)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/n/holyscratch01/mzitnik_lab/zaixizhang/BioLiP')
    parser.add_argument('--dest', type=str, default='/n/holyscratch01/mzitnik_lab/zaixizhang/BioLiP/split.pt')
    parser.add_argument('--train', type=int, default=100000)
    parser.add_argument('--val', type=int, default=1000)
    parser.add_argument('--test', type=int, default=20000)
    parser.add_argument('--val_num_pockets', type=int, default=-1)
    parser.add_argument('--test_num_pockets', type=int, default=100)
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()

    dataset = PocketLigandPairDataset(args.path)
    print('Load dataset successfully!')

    ids = torch.arange(len(dataset))
    train_id = ids[:-100]
    val_id = ids[-100:]
    test_id = ids[-100:]

    torch.save({
        'train': train_id,
        'val': val_id,
        'test': test_id,
    }, args.dest)

    print('Train %d, Validation %d, Test %d.' % (len(train_id), len(val_id), len(test_id)))
    print('Done.')
