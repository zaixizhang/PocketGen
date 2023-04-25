import torch
import numpy as np
import json, copy
import random
import glob
import csv
import os
import re
import Bio.PDB
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from bindgen.utils import full_square_dist

RESTYPE_1to3 = {
     "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN","E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
}

ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
ATOM_TYPES = [
    '', 'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
RES_ATOM14 = [
    [''] * 14,
    ['N', 'CA', 'C', 'O', 'CB', '',    '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'NE',  'CZ',  'NH1', 'NH2', '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'ND2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'OD1', 'OD2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'SG',  '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'NE2', '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'OE1', 'OE2', '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', '',   '',    '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'ND1', 'CD2', 'CE1', 'NE2', '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  'CE',  'NZ',  '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'SD',  'CE',  '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD',  '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG',  '',    '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    ['N', 'CA', 'C', 'O', 'CB', 'CG',  'CD1', 'CD2', 'CE1', 'CE2', 'CZ',  'OH',  '',    ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '',    '',    '',    '',    '',    '',    ''],
]


class AntibodyComplexDataset():

    def __init__(self, jsonl_file, cdr_type, L_target):
        self.data = []
        with open(jsonl_file) as f:
            all_lines = f.readlines()
            for line in tqdm(all_lines):
                entry = json.loads(line)
                assert len(entry['antibody_coords']) == len(entry['antibody_seq'])
                assert len(entry['antigen_coords']) == len(entry['antigen_seq'])
                if entry['antibody_cdr'].count(cdr_type) <= 4:
                    continue

                # paratope region
                surface = torch.tensor(
                        [i for i,v in enumerate(entry['antibody_cdr']) if v in cdr_type]
                )
                entry['binder_surface'] = surface

                entry['binder_seq'] = ''.join([entry['antibody_seq'][i] for i in surface.tolist()])
                entry['binder_coords'] = torch.tensor(entry['antibody_coords'])[surface]
                entry['binder_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['binder_seq']]
                )
                mask = (entry['binder_coords'].norm(dim=-1) > 1e-6).long()
                entry['binder_atypes'] *= mask

                # Create target
                entry['target_seq'] = entry['antigen_seq']
                entry['target_coords'] = torch.tensor(entry['antigen_coords'])
                entry['target_atypes'] = torch.tensor(
                        [[ATOM_TYPES.index(a) for a in RES_ATOM14[ALPHABET.index(s)]] for s in entry['target_seq']]
                )
                mask = (entry['target_coords'].norm(dim=-1) > 1e-6).long()
                entry['target_atypes'] *= mask

                # Find target surface
                dist, _ = full_square_dist(
                        entry['target_coords'][None,...], 
                        entry['binder_coords'][None,...], 
                        entry['target_atypes'][None,...], 
                        entry['binder_atypes'][None,...], 
                        contact=True
                )
                K = min(len(dist[0]), L_target)
                epitope = dist[0].amin(dim=-1).topk(k=K, largest=False).indices
                entry['target_surface'] = torch.sort(epitope).values

                if len(entry['binder_coords']) > 4 and len(entry['target_coords']) > 4 and entry['antibody_cdr'].count('001') <= 1:
                    self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ComplexLoader():

    def __init__(self, dataset, batch_tokens):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [len(dataset[i]['binder_seq']) for i in range(self.size)]
        self.batch_tokens = batch_tokens
        sorted_ix = np.argsort(self.lengths)

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        for ix in sorted_ix:
            size = self.lengths[ix]
            batch.append(ix)
            if size * (len(batch) + 1) > self.batch_tokens:
                clusters.append(batch)
                batch = []

        self.clusters = clusters
        if len(batch) > 0:
            clusters.append(batch)

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            batch = [self.dataset[i] for i in b_idx]
            yield batch


def make_batch_from_seq(batch):
    B = len(batch)
    L_max = max([len(seq) for seq in batch])
    S = np.zeros([B, L_max], dtype=np.int32)
    mask = np.zeros([B, L_max], dtype=np.float32)

    for i,seq in enumerate(batch):
        l = len(seq)
        indices = np.asarray([ALPHABET.index(a) for a in seq], dtype=np.int32)
        S[i, :l] = indices
        mask[i, :l] = 1.

    S = torch.from_numpy(S).long().to('cuda:1')
    mask = torch.from_numpy(mask).float().to('cuda:1')
    return S, mask


def featurize(batch, name):
    B = len(batch)
    L_max = max([len(b[name + "_seq"]) for b in batch])
    X = torch.zeros([B, L_max, 14, 3])
    S = torch.zeros([B, L_max]).long()
    A = torch.zeros([B, L_max, 14]).long()
    V = torch.zeros([B, L_max, 12])

    # Build the batch
    for i, b in enumerate(batch):
        l = len(b[name + '_seq'])
        X[i,:l] = b[name + '_coords']
        A[i,:l] = b[name + '_atypes']
        V[i,:l] = b[name + '_dihedrals'] if name + '_dihedrals' in b else 0
        indices = torch.tensor([ALPHABET.index(a) for a in b[name + '_seq']])
        S[i,:l] = indices

    return X.to('cuda:1'), S.to('cuda:1'), A.to('cuda:1'), V.to('cuda:1')


def make_batch(batch):
    target = featurize(batch, 'target')
    binder = featurize(batch, 'binder')
    surface = ([b['binder_surface'] for b in batch], [b['target_surface'] for b in batch])
    return binder, target, surface
