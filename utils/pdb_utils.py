#!/usr/bin/python
# -*- coding:utf-8 -*-
from copy import copy, deepcopy
import math
import os
from typing import Dict, List, Tuple
import requests

import numpy as np
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure as BStructure
from Bio.PDB.Model import Model as BModel
from Bio.PDB.Chain import Chain as BChain
from Bio.PDB.Residue import Residue as BResidue
from Bio.PDB.Atom import Atom as BAtom


class AminoAcid:
    def __init__(self, symbol: str, abrv: str, sidechain: List[str], idx=0):
        self.symbol = symbol
        self.abrv = abrv
        self.idx = idx
        self.sidechain = sidechain

    def __str__(self):
        return f'{self.idx} {self.symbol} {self.abrv} {self.sidechain}'


class AminoAcidVocab:

    MAX_ATOM_NUMBER = 14   # 4 backbone atoms + up to 10 sidechain atoms

    def __init__(self):
        self.backbone_atoms = ['N', 'CA', 'C', 'O']
        self.PAD, self.MASK = '#', '*'
        self.BOA, self.BOH, self.BOL = '&', '+', '-' # begin of antigen, heavy chain, light chain
        specials = [# special added
                (self.PAD, 'PAD'), (self.MASK, 'MASK'), # mask for masked / unknown residue
                (self.BOA, '<X>'), (self.BOH, '<H>'), (self.BOL, '<L>')
            ]

        aas = [
            ('A', 'ALA'), ('R', 'ARG'), ('N', 'ASN'), ('D', 'ASP'),
            ('C', 'CYS'), ('Q', 'GLN'), ('E', 'GLU'), ('G', 'GLY'),
            ('H', 'HIS'), ('I', 'ILE'), ('L', 'LEU'), ('K', 'LYS'),
            ('M', 'MET'), ('F', 'PHE'), ('P', 'PRO'), ('S', 'SER'),
            ('T', 'THR'), ('W', 'TRP'), ('Y', 'TYR'), ('V', 'VAL'),   # 20 aa
            # ('U', 'SEC') # 21 aa for eukaryote
        ]

        # max number of sidechain atoms: 10        
        self.atom_pad, self.atom_mask = 'p', 'm'
        self.atom_pos_mask, self.atom_pos_bb, self.atom_pos_pad = 'm', 'b', 'p'
        sidechain_map = {
            'G': [],   # -H
            'A': ['CB'],  # -CH3
            'V': ['CB', 'CG1', 'CG2'],  # -CH-(CH3)2
            'L': ['CB', 'CG', 'CD1', 'CD2'],  # -CH2-CH(CH3)2
            'I': ['CB', 'CG1', 'CG2', 'CD1'], # -CH(CH3)-CH2-CH3
            'F': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],  # -CH2-C6H5
            'W': ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],  # -CH2-C8NH6
            'Y': ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH'],  # -CH2-C6H4-OH
            'D': ['CB', 'CG', 'OD1', 'OD2'],  # -CH2-COOH
            'H': ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2'],  # -CH2-C3H3N2
            'N': ['CB', 'CG', 'OD1', 'ND2'],  # -CH2-CONH2
            'E': ['CB', 'CG', 'CD', 'OE1', 'OE2'],  # -(CH2)2-COOH
            'K': ['CB', 'CG', 'CD', 'CE', 'NZ'],  # -(CH2)4-NH2
            'Q': ['CB', 'CG', 'CD', 'OE1', 'NE2'],  # -(CH2)-CONH2
            'M': ['CB', 'CG', 'SD', 'CE'],  # -(CH2)2-S-CH3
            'R': ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2'],  # -(CH2)3-NHC(NH)NH2
            'S': ['CB', 'OG'],  # -CH2-OH
            'T': ['CB', 'OG1', 'CG2'],  # -CH(CH3)-OH
            'C': ['CB', 'SG'],  # -CH2-SH
            'P': ['CB', 'CG', 'CD'],  # -C3H6
        }

        self.chi_angles_atoms = {
            "ALA": [],
            # Chi5 in arginine is always 0 +- 5 degrees, so ignore it.
            "ARG": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "CD"],
                ["CB", "CG", "CD", "NE"],
                ["CG", "CD", "NE", "CZ"],
            ],
            "ASN": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
            "ASP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "OD1"]],
            "CYS": [["N", "CA", "CB", "SG"]],
            "GLN": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "CD"],
                ["CB", "CG", "CD", "OE1"],
            ],
            "GLU": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "CD"],
                ["CB", "CG", "CD", "OE1"],
            ],
            "GLY": [],
            "HIS": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "ND1"]],
            "ILE": [["N", "CA", "CB", "CG1"], ["CA", "CB", "CG1", "CD1"]],
            "LEU": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
            "LYS": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "CD"],
                ["CB", "CG", "CD", "CE"],
                ["CG", "CD", "CE", "NZ"],
            ],
            "MET": [
                ["N", "CA", "CB", "CG"],
                ["CA", "CB", "CG", "SD"],
                ["CB", "CG", "SD", "CE"],
            ],
            "PHE": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
            "PRO": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD"]],
            "SER": [["N", "CA", "CB", "OG"]],
            "THR": [["N", "CA", "CB", "OG1"]],
            "TRP": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
            "TYR": [["N", "CA", "CB", "CG"], ["CA", "CB", "CG", "CD1"]],
            "VAL": [["N", "CA", "CB", "CG1"]],
        }

        self.sidechain_bonds = {
            "ALA": { "CA": ["CB"] },
            "GLY": {},
            "VAL": {
                "CA": ["CB"],
                "CB": ["CG1", "CG2"]
            },
            "LEU": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD2", "CD1"]
            },
            "ILE": {
                "CA": ["CB"],
                "CB": ["CG1", "CG2"],
                "CG1": ["CD1"]
            },
            "MET": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["SD"],
                "SD": ["CE"],
            },
            "PHE": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD1", "CD2"],
                "CD1": ["CE1"],
                "CD2": ["CE2"],
                "CE1": ["CZ"]
            },
            "TRP": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD1", "CD2"],
                "CD1": ["NE1"],
                "CD2": ["CE2", "CE3"],
                "CE2": ["CZ2"],
                "CZ2": ["CH2"],
                "CE3": ["CZ3"]
            },
            "PRO": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"]
            },
            "SER": {
                "CA": ["CB"],
                "CB": ["OG"]
            },
            "THR": {
                "CA": ["CB"],
                "CB": ["OG1", "CG2"]
            },
            "CYS": {
                "CA": ["CB"],
                "CB": ["SG"]
            },
            "TYR": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD1", "CD2"],
                "CD1": ["CE1"],
                "CD2": ["CE2"],
                "CE1": ["CZ"],
                "CZ": ["OH"]
            },
            "ASN": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["OD1", "ND2"]
            },
            "GLN": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"],
                "CD": ["OE1", "NE2"]
            },
            "ASP": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["OD1", "OD2"]
            },
            "GLU": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"],
                "CD": ["OE1", "OE2"]
            },
            "LYS": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"],
                "CD": ["CE"],
                "CE": ["NZ"]
            },
            "ARG": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["CD"],
                "CD": ["NE"],
                "NE": ["CZ"],
                "CZ": ["NH1", "NH2"]
            },
            "HIS": {
                "CA": ["CB"],
                "CB": ["CG"],
                "CG": ["ND1", "CD2"],
                "ND1": ["CE1"],
                "CD2": ["NE2"]
            }
        }
        

        _all = aas + specials
        self.amino_acids = [AminoAcid(symbol, abrv, sidechain_map.get(symbol, [])) for symbol, abrv in _all]
        self.symbol2idx, self.abrv2idx = {}, {}
        for i, aa in enumerate(self.amino_acids):
            self.symbol2idx[aa.symbol] = i
            self.abrv2idx[aa.abrv] = i
            aa.idx = i
        self.special_mask = [0 for _ in aas] + [1 for _ in specials]

        # atom level vocab
        self.idx2atom = [self.atom_pad, self.atom_mask, 'C', 'N', 'O', 'S']
        self.idx2atom_pos = [self.atom_pos_pad, self.atom_pos_mask, self.atom_pos_bb, 'B', 'G', 'D', 'E', 'Z', 'H']
        self.atom2idx, self.atom_pos2idx = {}, {}
        for i, atom in enumerate(self.idx2atom):
            self.atom2idx[atom] = i
        for i, atom_pos in enumerate(self.idx2atom_pos):
            self.atom_pos2idx[atom_pos] = i
    
    def abrv_to_symbol(self, abrv):
        idx = self.abrv_to_idx(abrv)
        return None if idx is None else self.amino_acids[idx].symbol

    def symbol_to_abrv(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return None if idx is None else self.amino_acids[idx].abrv

    def abrv_to_idx(self, abrv):
        abrv = abrv.upper()
        return self.abrv2idx.get(abrv, None)

    def symbol_to_idx(self, symbol):
        symbol = symbol.upper()
        return self.symbol2idx.get(symbol, None)
    
    def idx_to_symbol(self, idx):
        return self.amino_acids[idx].symbol

    def idx_to_abrv(self, idx):
        return self.amino_acids[idx].abrv

    def get_pad_idx(self):
        return self.symbol_to_idx(self.PAD)

    def get_mask_idx(self):
        return self.symbol_to_idx(self.MASK)
    
    def get_special_mask(self):
        return copy(self.special_mask)

    def get_atom_type_mat(self):
        atom_pad = self.get_atom_pad_idx()
        mat = []
        for i, aa in enumerate(self.amino_acids):
            atoms = [atom_pad for _ in range(self.MAX_ATOM_NUMBER)]
            if aa.symbol == self.PAD:
                pass
            elif self.special_mask[i] == 1:  # specials
                atom_mask = self.get_atom_mask_idx()
                atoms = [atom_mask for _ in range(self.MAX_ATOM_NUMBER)]
            else:
                for aidx, atom in enumerate(self.backbone_atoms + aa.sidechain):
                    atoms[aidx] = self.atom_to_idx(atom[0])
            mat.append(atoms)
        return mat

    def get_atom_pos_mat(self):
        atom_pos_pad = self.get_atom_pos_pad_idx()
        mat = []
        for i, aa in enumerate(self.amino_acids):
            aps = [atom_pos_pad for _ in range(self.MAX_ATOM_NUMBER)]
            if aa.symbol == self.PAD:
                pass
            elif self.special_mask[i] == 1:
                atom_pos_mask = self.get_atom_pos_mask_idx()
                aps = [atom_pos_mask for _ in range(self.MAX_ATOM_NUMBER)]
            else:
                aidx = 0
                for _ in self.backbone_atoms:
                    aps[aidx] = self.atom_pos_to_idx(self.atom_pos_bb)
                    aidx += 1
                for atom in aa.sidechain:
                    aps[aidx] = self.atom_pos_to_idx(atom[1])
                    aidx += 1
            mat.append(aps)
        return mat

    def get_sidechain_info(self, symbol):
        idx = self.symbol_to_idx(symbol)
        return copy(self.amino_acids[idx].sidechain)
    
    def get_sidechain_geometry(self, symbol):
        abrv = self.symbol_to_abrv(symbol)
        chi_angles_atoms = copy(self.chi_angles_atoms[abrv])
        sidechain_bonds = self.sidechain_bonds[abrv]
        return (chi_angles_atoms, sidechain_bonds)
    
    def get_atom_pad_idx(self):
        return self.atom2idx[self.atom_pad]
    
    def get_atom_mask_idx(self):
        return self.atom2idx[self.atom_mask]
    
    def get_atom_pos_pad_idx(self):
        return self.atom_pos2idx[self.atom_pos_pad]

    def get_atom_pos_mask_idx(self):
        return self.atom_pos2idx[self.atom_pos_mask]
    
    def idx_to_atom(self, idx):
        return self.idx2atom[idx]

    def atom_to_idx(self, atom):
        return self.atom2idx[atom]

    def idx_to_atom_pos(self, idx):
        return self.idx2atom_pos[idx]
    
    def atom_pos_to_idx(self, atom_pos):
        return self.atom_pos2idx[atom_pos]

    def get_num_atom_type(self):
        return len(self.idx2atom)
    
    def get_num_atom_pos(self):
        return len(self.idx2atom_pos)

    def get_num_amino_acid_type(self):
        return len(self.special_mask) - sum(self.special_mask)

    def __len__(self):
        return len(self.symbol2idx)


VOCAB = AminoAcidVocab()


def format_aa_abrv(abrv):  # special cases
    if abrv == 'MSE':
        return 'MET' # substitue MSE with MET
    return abrv


def fetch_from_pdb(identifier):
    # example identifier: 1FBI
    url = 'https://data.rcsb.org/rest/v1/core/entry/' + identifier
    res = requests.get(url)
    if res.status_code != 200:
        return None
    url = f'https://files.rcsb.org/download/{identifier}.pdb'
    text = requests.get(url)
    data = res.json()
    data['pdb'] = text.text
    return data

