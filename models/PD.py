import copy
import sys

sys.path.append("..")
import numpy as np
from rdkit import RDConfig
import random
import os
import torch
import torch.nn as nn
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data, Batch
from rdkit.Chem import ChemicalFeatures
from rdkit import Chem
from rdkit.Chem import rdchem

from .encoders import get_encoder, MLP
from .encoders.cftfm import residue_atom_mask
from .common import *
from .protein_features import *
from .esmadapter import *
from .esm2adapter import *
from utils.pdb_utils import VOCAB
from utils.rmsd import kabsch_torch
from utils.protein_ligand import PDBProtein
from utils.relax import openmm_relax

ATOM_FAMILIES = ['Acceptor', 'Donor', 'Aromatic', 'Hydrophobe', 'LumpedHydrophobe', 'NegIonizable', 'PosIonizable',
                 'ZnBinder']
ATOM_FAMILIES_ID = {s: i for i, s in enumerate(ATOM_FAMILIES)}
NUM_ATOMS = [4, 5, 11, 8, 8, 6, 9, 9, 4, 10, 8, 8, 9, 8, 11, 7, 6, 7, 14, 12, 7]

ATOM_TYPES = [
    '', 'N', 'CA', 'C', 'O', 'CB', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
RES_ATOM14 = [
    [''] * 14,
    ['N', 'CA', 'C', 'O', 'CB', '', '', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'ND2', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'OD1', 'OD2', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'SG', '', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'NE2', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'OE1', 'OE2', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', '', '', '', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', 'CD1', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', 'CE', 'NZ', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG', '', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'OG1', 'CG2', '', '', '', '', '', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    ['N', 'CA', 'C', 'O', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'OH', '', ''],
    ['N', 'CA', 'C', 'O', 'CB', 'CG1', 'CG2', '', '', '', '', '', '', ''],
]

RES_ATOM_TYPE = [[ATOM_TYPES.index(a) for a in res]for res in RES_ATOM14]

AA_NUMBER_NAME = {1: 'ALA', 2: 'ARG', 3: 'ASN', 4: 'ASP', 5: 'CYS', 6: 'GLN', 7: 'GLU', 8: 'GLY', 9: 'HIS', 10: 'ILE',
                  11: 'LEU', 12: 'LYS', 13: 'MET', 14: 'PHE', 15: 'PRO', 16: 'SER', 17: 'THR', 18: 'TRP', 19: 'TYR',
                  20: 'VAL'}

RES_ATOMS = [[ATOM_TYPES.index(i) for i in res if i != ''] for res in RES_ATOM14]

BOND_TYPE = {1: rdchem.BondType.SINGLE, 2: rdchem.BondType.DOUBLE, 3: rdchem.BondType.TRIPLE,
             12: rdchem.BondType.AROMATIC}

def quaternion_to_matrix(q):
    """Convert a quaternion to its corresponding rotation matrix."""
    q = q / q.norm()
    w, x, y, z = q
    R = torch.zeros((3, 3), device=q.device)
    R[0, 0] = 1 - 2 * (y ** 2 + z ** 2)
    R[0, 1] = 2 * (x * y - z * w)
    R[0, 2] = 2 * (x * z + y * w)
    R[1, 0] = 2 * (x * y + z * w)
    R[1, 1] = 1 - 2 * (x ** 2 + z ** 2)
    R[1, 2] = 2 * (y * z - x * w)
    R[2, 0] = 2 * (x * z - y * w)
    R[2, 1] = 2 * (y * z + x * w)
    R[2, 2] = 1 - 2 * (x ** 2 + y ** 2)
    return R


def nearest(residue_mask):
    index = [[0, 0] for _ in range(len(residue_mask))]
    p, q = 0, len(residue_mask)
    for i in range(len(residue_mask)):
        if residue_mask[i] == 0:
            p = i
        else:
            index[i][0] = p
    for i in range(len(residue_mask) - 1, -1, -1):
        if residue_mask[i] == 0:
            q = i
        else:
            index[i][1] = q
    return index


def interpolation_init(pred_X, residue_mask, backbone_pos, atom2residue, protein_atom_batch, residue_batch):
    num_protein = protein_atom_batch.max().item() + 1
    offset = 0
    for i in range(num_protein):
        residue_mask_i = residue_mask[residue_batch == i]
        backbone_pos_i = backbone_pos[residue_batch == i]
        if (~residue_mask_i).sum() <= 2:
            offset += len(residue_mask_i)
            continue
        else:
            residue_index = torch.arange(len(residue_mask_i)).to(protein_atom_batch.device)
            front = residue_index[~residue_mask_i][:2]
            end = residue_index[~residue_mask_i][-2:]
            near = nearest(residue_mask_i)
            for k in range(len(residue_mask_i)):
                if residue_mask_i[k]:
                    mask = atom2residue == (k + offset)
                    if k < front[0]:
                        pred_X[mask] = backbone_pos_i[front[0]] + (k - front[0]) / (front[0] - front[1]) * (
                                    backbone_pos_i[front[0]] - backbone_pos_i[front[1]])
                    elif k > end[1]:
                        pred_X[mask] = backbone_pos_i[end[1]] + (k - end[1]) / (end[1] - end[0]) * (
                                    backbone_pos_i[end[1]] - backbone_pos_i[end[0]])
                    else:
                        pred_X[mask] = ((k - near[k][0]) * backbone_pos_i[near[k][1]] + (near[k][1] - k) *
                                        backbone_pos_i[near[k][0]]) * 1 / (near[k][1] - near[k][0])
            offset += len(residue_mask_i)

    return pred_X


def interpolation_init_new(res_X, residue_mask, backbone_pos, residue_batch):
    num_protein = residue_batch.max().item() + 1
    offset = 0
    backbone = torch.tensor([[-0.525, 1.363, 0.0], [0.0, 0.0, 0.0], [1.526, 0.0, 0.0], [0.627, 1.062, 0.0]],
                            device=res_X.device)
    for i in range(num_protein):
        residue_mask_i = residue_mask[residue_batch == i]
        backbone_pos_i = backbone_pos[residue_batch == i]
        if (~residue_mask_i).sum() <= 2:
            offset += len(residue_mask_i)
            continue
        else:
            residue_index = torch.arange(len(residue_mask_i)).to(res_X.device)
            front = residue_index[~residue_mask_i][:2]
            end = residue_index[~residue_mask_i][-2:]
            near = nearest(residue_mask_i)
            for k in range(len(residue_mask_i)):
                if residue_mask_i[k]:
                    ind = k + offset
                    if k < front[0]:
                        alpha = (backbone_pos_i[front[0]] + (k - front[0]) / (front[0] - front[1]) * (backbone_pos_i[front[0]] - backbone_pos_i[front[1]]))[1: 2]
                    elif k > end[1]:
                        alpha = (backbone_pos_i[end[1]] + (k - end[1]) / (end[1] - end[0]) * (backbone_pos_i[end[1]] - backbone_pos_i[end[0]]))[1: 2]
                    else:
                        alpha = (((k - near[k][0]) * backbone_pos_i[near[k][1]] + (near[k][1] - k) * backbone_pos_i[near[k][0]]) * 1 / (near[k][1] - near[k][0]))[1: 2]
                    res_X[ind][:4] = alpha + backbone @ quaternion_to_matrix(q=torch.randn(4, device=res_X.device)).t()
            offset += len(residue_mask_i)

    return res_X


class Pocket_Design_new(Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, device):
        super().__init__()
        self.config = config
        self.device = device
        self.hidden_channels = config.hidden_channels
        self.protein_atom_emb = nn.Embedding(protein_atom_feature_dim, int(config.hidden_channels/2-8))
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, config.hidden_channels)
        self.encoder = get_encoder(config.encoder, device)
        self.residue_mlp = Linear(config.hidden_channels, 20)
        self.Softmax = nn.Softmax(dim=1)
        self.huber_loss = nn.SmoothL1Loss(reduction='mean')
        self.dist_loss = torch.nn.MSELoss(reduction='mean')
        self.pred_loss = nn.CrossEntropyLoss(reduction='mean')
        self.interpolate_steps = 3
        self.atom_pos_embedding = nn.Embedding(14, 8)
        self.residue_embedding = nn.Embedding(21, int(config.hidden_channels/2 - 16))  # one embedding for mask
        self.standard2alphabet = torch.tensor([1, 6, 13, 9, 19, 12, 5, 2, 17, 8, 0, 11, 16, 14, 10, 4, 7, 18, 15, 3]).to(device)
        self.alphabet2standard = torch.tensor([10, 0, 7, 19, 15, 6, 1, 16, 9, 3, 14, 11, 5, 2, 13, 18, 12, 8, 17, 4]).to(device)
        self.residue_atom_mask = residue_atom_mask.to(device)
        self.write_pdb = True
        self.write_whole_pdb = False
        self.generate_id = 0
        self.generate_id1 = 0
        self.proteinloss = ProteinFeature()
        self.pe = PositionalEncodings(16)
        self.res_atom_type = torch.tensor(RES_ATOM_TYPE).to(device)
        self.orig_data_path = config.orig_data_path
        self.pocket10_path = config.pocket10_path
        if config.encoder.esm[:4] == 'esm2':
            encoder_args = {'_target_': 'esm2_adapter',
                            'encoder': {'d_model': 128,
                                        'use_esm_alphabet': True},
                            'dropout': 0.1,
                            'adapter_layer_indices': [6, 20, 32]}
            self.esmadapter = ESM2WithStructuralAdatper.from_pretrained(args=encoder_args, name=config.encoder.esm).to(device)
        else:
            encoder_args = {'_target_': 'esm_adapter',
                            'encoder': {'d_model': 128,
                                        'n_enc_layers': 3,
                                        'n_dec_layers': 3,
                                        'use_esm_alphabet': True},
                            'adapter_layer_indices': [6, 20, 32]}
            self.esmadapter = ProteinBertModelWithStructuralAdatper.from_pretrained(args=encoder_args).to(device)

    def forward_(self, batch):
        loss_list = [0., 0., 0.]
        residue_mask = batch['protein_edit_residue']
        full_seq = batch['seq']
        ligand_mask = batch['ligand_mask'].bool()
        label_ligand, pred_ligand = copy.deepcopy(batch['ligand_pos']), copy.deepcopy(batch['ligand_pos'])

        # init res_X
        label_X, res_X = copy.deepcopy(batch['residue_pos']), copy.deepcopy(batch['residue_pos'])
        res_X = interpolation_init_new(res_X, residue_mask, copy.deepcopy(batch['backbone_pos']),batch['amino_acid_batch'])
        for k in range(len(batch['amino_acid'])):
            if residue_mask[k]:
                pos = res_X[k]
                pos[4:] = (pos[1].repeat(10, 1) + 0.1 * torch.randn(10, 3, device=self.device))
                res_X[k] = pos
        pred_ligand = label_ligand + torch.randn_like(label_ligand).to(self.device) * 0.5

        ligand_feat = self.ligand_atom_emb(batch['ligand_feat'])

        for t in range(self.interpolate_steps):
            print(t)
            res_S = copy.deepcopy(batch['amino_acid_processed'])
            if t > 1:
                '''
                res_H[residue_mask] = res_H[residue_mask] + torch.matmul(pred_res_type[:, self.alphabet2standard].detach().float(), self.residue_embedding(torch.arange(1, 21).to(self.device))).unsqueeze(1)
                res_H[~residue_mask] = res_H[~residue_mask] + self.residue_embedding(res_S[~residue_mask]).unsqueeze(-2)
                '''
                res_S[residue_mask] = self.alphabet2standard[sampled_type.detach().clone()] + 1
                atom_emb = self.protein_atom_emb(self.res_atom_type[res_S])  # atom embedding
                atom_pos_emb = self.atom_pos_embedding(torch.arange(14).to(self.device)).unsqueeze(0).repeat(res_S.shape[0], 1, 1)  # pos embedding
                res_emb = self.residue_embedding(res_S).unsqueeze(-2).repeat(1, 14, 1)  # res embedding
                res_pos_emb = self.pe(batch['res_idx']).unsqueeze(-2).repeat(1, 14, 1)  # res pos embedding
                res_H = torch.cat([atom_emb, atom_pos_emb, res_emb, res_pos_emb], dim=-1)
            elif t <= 1:
                atom_emb = self.protein_atom_emb(self.res_atom_type[res_S])  # atom embedding
                atom_pos_emb = self.atom_pos_embedding(torch.arange(14).to(self.device)).unsqueeze(0).repeat(res_S.shape[0], 1, 1)  # pos embedding
                res_emb = self.residue_embedding(res_S).unsqueeze(-2).repeat(1, 14, 1)  # res embedding
                res_pos_emb = self.pe(batch['res_idx']).unsqueeze(-2).repeat(1, 14, 1)  # res pos embedding
                res_H = torch.cat([atom_emb, atom_pos_emb, res_emb, res_pos_emb], dim=-1)

            _, res_X, pred_res_type, pred_ligand = self.encoder(res_H, res_X.detach().clone(), res_S, batch['amino_acid_batch'], full_seq, pred_ligand.detach().clone(),
                             ligand_feat, batch['ligand_mask'], batch['edit_residue_num'], residue_mask, self.esmadapter, batch['full_seq_mask'], batch['r10_mask'])

            atom_mask = self.residue_atom_mask[batch['amino_acid'][residue_mask]].bool()

            loss_list[0] += 2*self.huber_loss(res_X[residue_mask][atom_mask],label_X[residue_mask][atom_mask]) + self.huber_loss(pred_ligand[ligand_mask], label_ligand[ligand_mask])
            loss_list[1] += self.pred_loss(pred_res_type, self.standard2alphabet[batch['amino_acid'][residue_mask] - 1])
            # bond and angle loss
            # loss_list[2] += 3*self.proteinloss.structure_loss(res_X[residue_mask], label_X[residue_mask], batch['amino_acid'][residue_mask] - 1, batch['res_idx'][residue_mask], batch['amino_acid_batch'][residue_mask])
            loss_list[2] += 0.

            sampled_type, _ = sample_from_categorical(pred_res_type.detach())
        aar = (self.standard2alphabet[batch['amino_acid'][residue_mask] - 1] == sampled_type).sum() / len(res_S[residue_mask])
        rmsd = torch.sqrt((res_X[residue_mask][:, :4].reshape(-1, 3) - label_X[residue_mask][:, :4].reshape(-1, 3)).norm(dim=1).sum() / len(res_S[residue_mask]) / 4)

        return loss_list[1] + loss_list[0] + loss_list[2], loss_list, aar, rmsd

    def init(self, batch):
        residue_mask = batch['protein_edit_residue']
        label_ligand, pred_ligand = copy.deepcopy(batch['ligand_pos']), copy.deepcopy(batch['ligand_pos'])
        pred_ligand = label_ligand + torch.randn_like(label_ligand).to(self.device) * 0.5
        res_X = copy.deepcopy(batch['residue_pos'])  # init res_X
        res_X = interpolation_init_new(res_X, residue_mask, copy.deepcopy(batch['backbone_pos']),
                                       batch['amino_acid_batch'])
        res_S = copy.deepcopy(batch['amino_acid_processed'])
        for k in range(len(batch['amino_acid'])):  # init side chain atoms of masked residues
            if residue_mask[k]:
                pos = res_X[k]
                pos[4:] = (pos[1].repeat(10, 1) + 0.1 * torch.randn(10, 3, device=self.device))
                res_X[k] = pos

        ligand_feat = self.ligand_atom_emb(batch['ligand_feat'])

        atom_emb = self.protein_atom_emb(self.res_atom_type[res_S])  # atom embedding
        atom_pos_emb = self.atom_pos_embedding(torch.arange(14).to(self.device)).unsqueeze(0).repeat(res_S.shape[0], 1,
                                                                                                     1)  # pos embedding
        res_emb = self.residue_embedding(res_S).unsqueeze(-2).repeat(1, 14, 1)  # res embedding
        res_pos_emb = self.pe(batch['res_idx']).unsqueeze(-2).repeat(1, 14, 1)  # res pos embedding
        res_H = torch.cat([atom_emb, atom_pos_emb, res_emb, res_pos_emb], dim=-1)
        self.seq = batch['seq']
        self.full_seq_mask = batch['full_seq_mask']
        self.r10_mask = batch['r10_mask']
        return res_H, res_X, res_S, batch['amino_acid_batch'], pred_ligand, ligand_feat, batch['ligand_mask'], batch['edit_residue_num'], residue_mask

    def forward(self, res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask, use_esm=True):
        '''
        res_H[residue_mask] = res_H[residue_mask] + torch.matmul(pred_res_type[:, self.alphabet2standard].detach().float(), self.residue_embedding(torch.arange(1, 21).to(self.device))).unsqueeze(1)
        res_H[~residue_mask] = res_H[~residue_mask] + self.residue_embedding(res_S[~residue_mask]).unsqueeze(-2)
        '''

        res_H, res_X, ligand_pos, ligand_feat, pred_res_type = self.encoder(res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask)

        if use_esm and self.seq.shape[1] <= 1000:
            h_residue = res_H.sum(-2)
            batch_size = res_batch.max().item() + 1
            encoder_out = {'feats': torch.zeros(batch_size, self.seq.shape[1], self.hidden_channels).to(self.device)}
            encoder_out['feats'][self.r10_mask] = h_residue.view(-1, self.hidden_channels)
            init_pred = self.seq
            decode_logits = self.esmadapter(init_pred, encoder_out)['logits']
            pred_res_type = decode_logits[self.full_seq_mask][:, 4:24]

        return res_H, res_X, ligand_pos, ligand_feat, pred_res_type

    def generate(self, batch, target_path='./generate'):
        print('Start Generating')
        residue_mask = batch['protein_edit_residue']
        res_S = batch['amino_acid_processed']
        full_seq = batch['seq']
        label_S = copy.deepcopy(batch['amino_acid'])
        label_X, res_X = copy.deepcopy(batch['residue_pos']), copy.deepcopy(batch['residue_pos'])
        label_ligand, pred_ligand = copy.deepcopy(batch['ligand_pos']), copy.deepcopy(batch['ligand_pos'])
        res_X = interpolation_init_new(res_X, residue_mask, copy.deepcopy(batch['backbone_pos']), batch['amino_acid_batch'])
        res_batch = batch['amino_acid_batch']
        for k in range(len(batch['amino_acid'])):
            if residue_mask[k]:
                pos = res_X[k]
                pos[4:] = (pos[1].repeat(10, 1) + 0.1 * torch.randn(10, 3, device=self.device))
                res_X[k] = pos

        ligand_feat = self.ligand_atom_emb(batch['ligand_feat'])

        for t in range(self.interpolate_steps):
            if t < -1:
                res_S[residue_mask] = self.alphabet2standard[sampled_type.detach().clone()] + 1
                atom_emb = self.protein_atom_emb(self.res_atom_type[res_S])  # atom embedding
                atom_pos_emb = self.atom_pos_embedding(torch.arange(14).to(self.device)).unsqueeze(0).repeat(res_S.shape[0], 1, 1)  # pos embedding
                res_emb = self.residue_embedding(res_S).unsqueeze(-2).repeat(1, 14, 1)  # res embedding
                res_pos_emb = self.pe(batch['res_idx']).unsqueeze(-2).repeat(1, 14, 1)  # res pos embedding
                res_H = torch.cat([atom_emb, atom_pos_emb, res_emb, res_pos_emb], dim=-1)
            elif t == 0:
                atom_emb = self.protein_atom_emb(self.res_atom_type[res_S])  # atom embedding
                atom_pos_emb = self.atom_pos_embedding(torch.arange(14).to(self.device)).unsqueeze(0).repeat(res_S.shape[0], 1, 1)  # pos embedding
                res_emb = self.residue_embedding(res_S).unsqueeze(-2).repeat(1, 14, 1)  # res embedding
                res_pos_emb = self.pe(batch['res_idx']).unsqueeze(-2).repeat(1, 14, 1)  # res pos embedding
                res_H = torch.cat([atom_emb, atom_pos_emb, res_emb, res_pos_emb], dim=-1)

            res_H, res_X, pred_ligand, ligand_feat, pred_res_type = self.encoder(res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, batch['ligand_mask'], batch['edit_residue_num'], residue_mask)
            if full_seq.shape[1] <= 1000:
                h_residue = res_H.sum(-2)
                batch_size = res_batch.max().item() + 1
                encoder_out = {
                    'feats': torch.zeros(batch_size, full_seq.shape[1], self.hidden_channels).to(self.device)}
                encoder_out['feats'][batch['r10_mask']] = h_residue.view(-1, self.hidden_channels)
                init_pred = full_seq
                decode_logits = self.esmadapter(init_pred, encoder_out)['logits']
                pred_res_type = decode_logits[batch['full_seq_mask']][:, 4:24]

            sampled_type, _ = sample_from_categorical(pred_res_type)

        aar = (self.standard2alphabet[batch['amino_acid'][residue_mask] - 1] == sampled_type).sum() / len(label_S[residue_mask])
        rmsd = torch.sqrt((res_X[residue_mask][:, :4].reshape(-1, 3) - label_X[residue_mask][:, :4].reshape(-1, 3)).norm(dim=1).sum() / len(label_S[residue_mask]) / 4)
        
        if self.write_pdb:
            res_S[residue_mask] = self.alphabet2standard[sampled_type.detach().clone()] + 1
            to_sdf(pred_ligand, batch['ligand_element'].long(), batch['ligand_mask'].bool(), batch['ligand_batch'],batch['ligand_bond_type'].long(), batch['ligand_bond_index'].long(), batch['edge_batch'], self.generate_id, target_path)
            to_pdb(label_X, batch['amino_acid'], batch['res_idx'], batch['amino_acid_batch'], self.generate_id, batch['protein_filename'], target_path, original=True)
            self.generate_id = to_pdb(res_X, res_S, batch['res_idx'], batch['amino_acid_batch'], self.generate_id, batch['protein_filename'], target_path, original = False)
            
        if self.write_whole_pdb:
            self.generate_id1 = to_whole_pdb(res_X, res_S, batch['res_idx'], batch['amino_acid_batch'], self.generate_id1, batch['protein_filename'], batch['r10_mask'], self.orig_data_path, target_path)
        return aar, rmsd


def sample_from_categorical(logits=None, temperature=2.0):
    if temperature:
        dist = torch.distributions.Categorical(logits=logits.div(temperature))
        tokens = dist.sample()
        scores = dist.log_prob(tokens)
    else:
        scores, tokens = logits.log_softmax(dim=-1).max(dim=-1)
    return tokens, scores


def sample_from_topk(tensor, k=3):
    """
    Apply softmax to the tensor, then randomly sample an index from the top k values.

    :param tensor: Input tensor.
    :param k: Number of top values to consider.
    :return: Index of the sampled value.
    """

    # Apply softmax
    probs = torch.nn.functional.softmax(tensor, dim=0)

    # Get top k values and their indices
    _, top_indices = torch.topk(probs, k)

    sampled_indices = torch.randint(0, k, (top_indices.shape[0],))

    # Use advanced indexing to gather the sampled elements from each row
    sampled_elements = top_indices[torch.arange(top_indices.shape[0]), sampled_indices]

    return sampled_elements, None


def random_mask(batch, device, mask=True):
    if mask:
        tmp = []
        num_protein = batch['protein_atom_batch'].max() + 1
        for i in range(num_protein):
            mask = batch['amino_acid_batch'] == i
            ind = torch.multinomial(batch['protein_edit_residue'][mask].float(), 1)
            selected = torch.zeros_like(batch['protein_edit_residue'][mask], dtype=bool)
            selected[ind] = 1
            tmp.append(selected)
        batch['random_mask_residue'] = torch.cat(tmp, dim=0)

        # remove side chains for the masked atoms
        index = torch.arange(len(batch['amino_acid']))[batch['random_mask_residue']]
        for key in ['protein_pos', 'protein_atom_feature']:
            tmp = []
            for k in range(batch['atom2residue'].max() + 1):
                mask = batch['atom2residue'] == k
                if k in index:
                    if key == 'protein_atom_feature':
                        feature_mask = batch['protein_atom_feature'][mask]
                        feature_mask[:, -20:] = torch.zeros(20, device=device)
                        feature_mask[:, -21] = 1
                        batch['protein_atom_feature'][mask] = feature_mask
                    tmp.append(batch[key][mask][:4])
                else:
                    tmp.append(batch[key][mask])
            batch[key] = torch.cat(tmp, dim=0)
        batch['residue_natoms'][batch['random_mask_residue']] = 4
        batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(batch['residue_natoms']), device=device),
                                                        batch['residue_natoms'])
        batch['protein_edit_atom'] = torch.repeat_interleave(batch['protein_edit_residue'], batch['residue_natoms'],
                                                             dim=0)
        batch['random_mask_atom'] = torch.repeat_interleave(batch['random_mask_residue'], batch['residue_natoms'],
                                                            dim=0)
    else:
        # reset protein pos and feature
        index = torch.arange(len(batch['amino_acid']))[batch['random_mask_residue']]
        num_residues = batch['atom2residue'].max() + 1
        pos_tmp, feature_tmp, natoms_tmp = [], [], []
        for k in range(num_residues):
            mask = batch['atom2residue'] == k
            res_type = batch['amino_acid'][k]
            sidechain_size = NUM_ATOMS[res_type] - 4
            if k in index:
                pos_tmp.append(batch['protein_pos'][mask][:4])
                if sidechain_size > 0:
                    pos_tmp.append(
                        batch['protein_pos'][mask][1:2].repeat(sidechain_size, 1) + 0.1 * torch.randn(sidechain_size, 3,
                                                                                                      device=device))
                feature_tmp.append(atom_feature(res_type, device))
                natoms_tmp.append(NUM_ATOMS[res_type])
            else:
                pos_tmp.append(batch['protein_pos'][mask])
                feature_tmp.append(batch['protein_atom_feature'][mask])
                natoms_tmp.append(batch['protein_pos'][mask].shape[0])
        batch['protein_pos'], batch['protein_atom_feature'] = torch.cat(pos_tmp, dim=0), torch.cat(feature_tmp, dim=0)
        batch['protein_atom_feature'][:, -21] = 0

        batch['residue_natoms'] = torch.tensor(natoms_tmp, device=device)
        batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(batch['residue_natoms']), device=device),
                                                        batch['residue_natoms'])
        batch['protein_edit_atom'] = torch.repeat_interleave(batch['protein_edit_residue'], batch['residue_natoms'],
                                                             dim=0)

    # follow batch
    num_protein = batch['protein_atom_batch'].max() + 1
    repeats = torch.tensor([batch['residue_natoms'][batch['amino_acid_batch'] == i].sum() for i in range(num_protein)])
    batch['protein_atom_batch'] = torch.repeat_interleave(torch.arange(num_protein), repeats).to(device)
    batch['edit_backbone'] = copy.deepcopy(batch['protein_edit_atom'])
    index = torch.arange(len(batch['amino_acid']))[batch['protein_edit_residue']]
    for k in range(len(batch['amino_acid'])):
        mask = batch['atom2residue'] == k
        if k in index:
            data_mask = batch['edit_backbone'][mask]
            data_mask[4:] = 0
            batch['edit_backbone'][mask] = data_mask

    return batch


def atom_feature(res_type, device):
    atom_types = torch.arange(38)
    max_num_aa = 21
    atom_type = torch.tensor(RES_ATOMS[res_type]).view(-1, 1) == atom_types.view(1, -1)
    amino_acid = F.one_hot(res_type, num_classes=max_num_aa).repeat(NUM_ATOMS[res_type], 1)
    x = torch.cat([atom_type.to(device), amino_acid], dim=-1)
    return x


def to_pdb(res_X, amino_acid, res_idx, res_batch, index, pocket_filename, target_path, original):
    lines = ['HEADER    POCKET', 'COMPND    POCKET\n']
    num_protein = res_batch.max().item() + 1
    for n in range(num_protein):
        #pdb_path = os.path.join(orig_data_path, pocket_filename[n])
        pdb_path = pocket_filename[n]
        with open(pdb_path, 'r') as f:
            pdb_block = f.read()
        protein = PDBProtein(pdb_block)
        residues, atoms = protein.return_residues()
        mask = (res_batch == n)
        res_X_protein = res_X[mask]
        amino_acid_protein = amino_acid[mask]
        res_idx_protein = res_idx[mask]
        atom_count = 0
        if original:
            path = os.path.join(target_path, str(index + n) + '_orig.pdb')
        else:
            path = os.path.join(target_path, str(index + n) + '.pdb')
        with open(path, 'w') as f:
            f.writelines(lines)
            for k in range(len(res_X_protein)):
                atom_type = RES_ATOM14[amino_acid_protein[k]]
                chain = residues[k]['chain']
                for i in range(NUM_ATOMS[amino_acid_protein[k]]):
                    j0 = str('ATOM').ljust(6)  # atom#6s
                    j1 = str(atom_count).rjust(5)  # aomnum#5d
                    j2 = str(atom_type[i]).center(4)  # atomname$#4s
                    j3 = AA_NUMBER_NAME[amino_acid_protein[k].item()].ljust(3)  # resname#1s
                    j4 = str(chain).rjust(1)  # Astring
                    j5 = str(res_idx_protein[k].item()).rjust(4)  # resnum
                    j6 = str('%8.3f' % (float(res_X_protein[k, i, 0]))).rjust(8)  # x
                    j7 = str('%8.3f' % (float(res_X_protein[k, i, 1]))).rjust(8)  # y
                    j8 = str('%8.3f' % (float(res_X_protein[k, i, 2]))).rjust(8)  # z\
                    j9 = str('%6.2f' % (1.00)).rjust(6)  # occ
                    j10 = str('%6.2f' % (25.02)).ljust(6)  # temp
                    j11 = str(atom_type[i][0]).rjust(12)  # elname
                    f.write("%s%s %s %s %s%s    %s%s%s%s%s%s\n" % (j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11))
                    atom_count += 1
            f.write('END')
            f.write('\n')
        openmm_relax(path)
    return index + num_protein


def to_whole_pdb(res_X, amino_acid, res_idx, res_batch, index, protein_filename, r10_mask, orig_data_path, target_path):
    lines = ['HEADER    POCKET', 'COMPND    POCKET\n']
    num_protein = res_batch.max().item() + 1
    for n in range(num_protein):
        pdb_path = orig_data_path + protein_filename[n]
        with open(pdb_path, 'r') as f:
            pdb_block = f.read()
        protein = PDBProtein(pdb_block)
        residues, atoms = protein.return_residues()

        mask = (res_batch == n)
        res_X_protein = res_X[mask]
        amino_acid_protein = amino_acid[mask]
        res_idx_protein = res_idx[mask]
        assert r10_mask[n].sum() == len(amino_acid_protein)

        path = target_path + str(index + n) + '_whole.pdb'
        atom_count = 0
        stored_res_count = 0
        with open(path, 'w') as f:
            f.writelines(lines)
            for k in range(len(residues)):
                if r10_mask[n, k+1]:
                    chain = atoms[residues[k]['atoms'][0]]['line'][21:22].strip()
                    atom_type = RES_ATOM14[amino_acid_protein[stored_res_count]]
                    for i in range(NUM_ATOMS[amino_acid_protein[stored_res_count]]):
                        j0 = str('ATOM').ljust(6)  # atom#6s
                        j1 = str(atom_count).rjust(5)  # aomnum#5d
                        j2 = str(atom_type[i]).center(4)  # atomname$#4s
                        j3 = AA_NUMBER_NAME[amino_acid_protein[stored_res_count].item()].ljust(3)  # resname#1s
                        j4 = str(chain).rjust(1)  # Astring
                        j5 = str(res_idx_protein[stored_res_count].item()).rjust(4)  # resnum
                        j6 = str('%8.3f' % (float(res_X_protein[stored_res_count, i, 0]))).rjust(8)  # x
                        j7 = str('%8.3f' % (float(res_X_protein[stored_res_count, i, 1]))).rjust(8)  # y
                        j8 = str('%8.3f' % (float(res_X_protein[stored_res_count, i, 2]))).rjust(8)  # z\
                        j9 = str('%6.2f' % (1.00)).rjust(6)  # occ
                        j10 = str('%6.2f' % (25.02)).ljust(6)  # temp
                        j11 = str(atom_type[i][0]).rjust(12)  # elname
                        f.write("%s%s %s %s %s%s    %s%s%s%s%s%s\n" % (j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11))
                        atom_count += 1
                    stored_res_count += 1
                else:
                    for atom_idx in residues[k]['atoms']:
                        line = atoms[atom_idx]['line']
                        line = line[:6] + str(atom_count).rjust(5) + line[11:] + "\n"
                        atom_count += 1
                        f.write(line)
            f.write('END')
            f.write('\n')
        openmm_relax(path)
    return index + num_protein


def to_sdf(pred_pos, elements, mask, ligand_batch, bond_types, bond_index, edge_batch, id, target_path):
    num_ligand = edge_batch.max().item() + 1
    for l in range(num_ligand):
        filename = os.path.join(target_path, str(id + l) + '.sdf')
        positions = pred_pos[l][mask[l]]
        elements_protein = elements[ligand_batch == l]
        bond_types_protein = bond_types[edge_batch == l]
        bond_index_protein = bond_index[:, edge_batch == l].transpose(0, 1)

        mol = rdchem.EditableMol(Chem.Mol())

        # Add atoms to molecule
        for element in elements_protein:
            atom = Chem.Atom(element.item())
            mol.AddAtom(atom)

        # Add bonds to molecule
        edge_set = set()
        for k, (bond_type, (start_idx, end_idx)) in enumerate(zip(bond_types_protein, bond_index_protein)):
            if (start_idx.item(), end_idx.item()) not in edge_set:
                edge_set.add((start_idx.item(), end_idx.item()))
                edge_set.add((end_idx.item(), start_idx.item()))
                mol.AddBond(start_idx.item(), end_idx.item(), BOND_TYPE[bond_type.item()])

        # Set 3D coordinates (assuming positions are in 3D)
        mol = mol.GetMol()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, position in enumerate(positions):
            conf.SetAtomPosition(i, position.tolist())
        mol.AddConformer(conf)

        writer = Chem.SDWriter(filename)
        writer.write(mol)
        writer.close()

    return mol


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)


class AminoAcidFeature(nn.Module):
    def __init__(self, backbone_only=False) -> None:
        super().__init__()

        self.backbone_only = backbone_only

        # number of classes
        self.num_aa_type = len(VOCAB)
        self.num_atom_type = VOCAB.get_num_atom_type()
        self.num_atom_pos = VOCAB.get_num_atom_pos()

        # atom-level special tokens
        self.atom_mask_idx = VOCAB.get_atom_mask_idx()
        self.atom_pad_idx = VOCAB.get_atom_pad_idx()
        self.atom_pos_mask_idx = VOCAB.get_atom_pos_mask_idx()
        self.atom_pos_pad_idx = VOCAB.get_atom_pos_pad_idx()

        # global nodes and mask nodes
        self.boa_idx = VOCAB.symbol_to_idx(VOCAB.BOA)
        self.boh_idx = VOCAB.symbol_to_idx(VOCAB.BOH)
        self.bol_idx = VOCAB.symbol_to_idx(VOCAB.BOL)
        self.mask_idx = VOCAB.get_mask_idx()

        # atoms encoding
        residue_atom_type, residue_atom_pos = [], []
        backbone = [VOCAB.atom_to_idx(atom[0]) for atom in VOCAB.backbone_atoms]
        n_channel = VOCAB.MAX_ATOM_NUMBER if not backbone_only else 4
        special_mask = VOCAB.get_special_mask()
        for i in range(len(VOCAB)):
            if i == self.boa_idx or i == self.boh_idx or i == self.bol_idx or i == self.mask_idx:
                # global nodes
                residue_atom_type.append([self.atom_mask_idx for _ in range(n_channel)])
                residue_atom_pos.append([self.atom_pos_mask_idx for _ in range(n_channel)])
            elif special_mask[i] == 1:
                # other special token (pad)
                residue_atom_type.append([self.atom_pad_idx for _ in range(n_channel)])
                residue_atom_pos.append([self.atom_pos_pad_idx for _ in range(n_channel)])
            else:
                # normal amino acids
                sidechain_atoms = VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i))
                atom_type = backbone
                atom_pos = [VOCAB.atom_pos_to_idx(VOCAB.atom_pos_bb) for _ in backbone]
                if not backbone_only:
                    sidechain_atoms = VOCAB.get_sidechain_info(VOCAB.idx_to_symbol(i))
                    atom_type = atom_type + [VOCAB.atom_to_idx(atom[0]) for atom in sidechain_atoms]
                    atom_pos = atom_pos + [VOCAB.atom_pos_to_idx(atom[1]) for atom in sidechain_atoms]
                num_pad = n_channel - len(atom_type)
                residue_atom_type.append(atom_type + [self.atom_pad_idx for _ in range(num_pad)])
                residue_atom_pos.append(atom_pos + [self.atom_pos_pad_idx for _ in range(num_pad)])

        # mapping from residue to atom types and positions
        self.residue_atom_type = nn.parameter.Parameter(
            torch.tensor(residue_atom_type, dtype=torch.long),
            requires_grad=False)
        self.residue_atom_pos = nn.parameter.Parameter(
            torch.tensor(residue_atom_pos, dtype=torch.long),
            requires_grad=False)

        # sidechain geometry
        if not backbone_only:
            sc_bonds, sc_bonds_mask = [], []
            sc_chi_atoms, sc_chi_atoms_mask = [], []
            for i in range(len(VOCAB)):
                if special_mask[i] == 1:
                    sc_bonds.append([])
                    sc_chi_atoms.append([])
                else:
                    symbol = VOCAB.idx_to_symbol(i)
                    atom_type = VOCAB.backbone_atoms + VOCAB.get_sidechain_info(symbol)
                    atom2channel = {atom: i for i, atom in enumerate(atom_type)}
                    chi_atoms, bond_atoms = VOCAB.get_sidechain_geometry(symbol)
                    sc_chi_atoms.append(
                        [[atom2channel[atom] for atom in atoms] for atoms in chi_atoms]
                    )
                    bonds = []
                    for src_atom in bond_atoms:
                        for dst_atom in bond_atoms[src_atom]:
                            bonds.append((atom2channel[src_atom], atom2channel[dst_atom]))
                    sc_bonds.append(bonds)
            max_num_chis = max([len(chis) for chis in sc_chi_atoms])
            max_num_bonds = max([len(bonds) for bonds in sc_bonds])
            for i in range(len(VOCAB)):
                num_chis, num_bonds = len(sc_chi_atoms[i]), len(sc_bonds[i])
                num_pad_chis, num_pad_bonds = max_num_chis - num_chis, max_num_bonds - num_bonds
                sc_chi_atoms_mask.append(
                    [1 for _ in range(num_chis)] + [0 for _ in range(num_pad_chis)]
                )
                sc_bonds_mask.append(
                    [1 for _ in range(num_bonds)] + [0 for _ in range(num_pad_bonds)]
                )
                sc_chi_atoms[i].extend([[-1, -1, -1, -1] for _ in range(num_pad_chis)])
                sc_bonds[i].extend([(-1, -1) for _ in range(num_pad_bonds)])

            # mapping residues to their sidechain chi angle atoms and bonds
            self.sidechain_chi_angle_atoms = nn.parameter.Parameter(
                torch.tensor(sc_chi_atoms, dtype=torch.long),
                requires_grad=False)
            self.sidechain_chi_mask = nn.parameter.Parameter(
                torch.tensor(sc_chi_atoms_mask, dtype=torch.bool),
                requires_grad=False
            )
            self.sidechain_bonds = nn.parameter.Parameter(
                torch.tensor(sc_bonds, dtype=torch.long),
                requires_grad=False
            )
            self.sidechain_bonds_mask = nn.parameter.Parameter(
                torch.tensor(sc_bonds_mask, dtype=torch.bool),
                requires_grad=False
            )

    def _construct_residue_pos(self, S):
        # construct residue position. global node is 1, the first residue is 2, ... (0 for padding)
        glbl_node_mask = self._is_global(S)
        glbl_node_idx = torch.nonzero(glbl_node_mask).flatten()  # [batch_size * 3] (boa, boh, bol)
        shift = F.pad(glbl_node_idx[:-1] - glbl_node_idx[1:] + 1, (1, 0), value=1)  # [batch_size * 3]
        residue_pos = torch.ones_like(S)
        residue_pos[glbl_node_mask] = shift
        residue_pos = torch.cumsum(residue_pos, dim=0)
        return residue_pos

    def _construct_segment_ids(self, res_idx, batch):
        consecutive = (res_idx[1:] == res_idx[:-1]) & (batch[1:] == batch[:-1])
        segment_ids = torch.zeros_like(res_idx).long()
        id = 0
        for i in range(1, len(segment_ids)):
            if consecutive[i - 1]:
                segment_ids[i] = id
            else:
                id += 1
                segment_ids[i] = id
        return segment_ids

    def _construct_atom_type(self, S):
        # construct atom types
        return self.residue_atom_type[S]

    def _construct_atom_pos(self, S):
        # construct atom positions
        return self.residue_atom_pos[S]

    @torch.no_grad()
    def get_sidechain_chi_angles_atoms(self, S):
        chi_angles_atoms = self.sidechain_chi_angle_atoms[S]  # [N, max_num_chis, 4]
        chi_mask = self.sidechain_chi_mask[S]  # [N, max_num_chis]
        return chi_angles_atoms, chi_mask

    @torch.no_grad()
    def get_sidechain_bonds(self, S):
        bonds = self.sidechain_bonds[S]  # [N, max_num_bond, 2]
        bond_mask = self.sidechain_bonds_mask[S]
        return bonds, bond_mask

    def forward(self, X, S, batch_id, k_neighbors):
        H, (_, _, atom_pos) = self.embedding(S)
        ctx_edges, inter_edges = self.construct_edges(
            X, S, batch_id, k_neighbors, atom_pos=atom_pos)
        return H, (ctx_edges, inter_edges)


class ProteinFeature(nn.Module):
    def __init__(self, backbone_only=False):
        super().__init__()
        self.backbone_only = backbone_only
        self.aa_feature = AminoAcidFeature()

    def _cal_sidechain_bond_lengths(self, S, X):
        bonds, bonds_mask = self.aa_feature.get_sidechain_bonds(S)
        n = torch.nonzero(bonds_mask)[:, 0]  # [Nbonds]
        src, dst = bonds[bonds_mask].T
        src_X, dst_X = X[(n, src)], X[(n, dst)]  # [Nbonds, 3]
        bond_lengths = torch.norm(dst_X - src_X, dim=-1)
        return bond_lengths

    def _cal_sidechain_chis(self, S, X):
        chi_atoms, chi_mask = self.aa_feature.get_sidechain_chi_angles_atoms(S)
        n = torch.nonzero(chi_mask)[:, 0]  # [Nchis]
        a0, a1, a2, a3 = chi_atoms[chi_mask].T  # [Nchis]
        x0, x1, x2, x3 = X[(n, a0)], X[(n, a1)], X[(n, a2)], X[(n, a3)]  # [Nchis, 3]
        u_0, u_1, u_2 = (x1 - x0), (x2 - x1), (x3 - x2)  # [Nchis, 3]
        # normals of the two planes
        n_1 = F.normalize(torch.cross(u_0, u_1), dim=-1)  # [Nchis, 3]
        n_2 = F.normalize(torch.cross(u_1, u_2), dim=-1)  # [Nchis, 3]
        cosChi = (n_1 * n_2).sum(-1)  # [Nchis]
        eps = 1e-7
        cosChi = torch.clamp(cosChi, -1 + eps, 1 - eps)
        return cosChi

    def _cal_backbone_bond_lengths(self, X, seg_id):
        # loss of backbone (...N-CA-C(O)-N...) bond length
        # N-CA, CA-C, C=O
        bl1 = torch.norm(X[:, 1:4] - X[:, :3], dim=-1)  # [N, 3], (N-CA), (CA-C), (C=O)
        # C-N
        bl2 = torch.norm(X[1:, 0] - X[:-1, 2], dim=-1)  # [N-1]
        same_chain_mask = seg_id[1:] == seg_id[:-1]
        bl2 = bl2[same_chain_mask]
        bl = torch.cat([bl1.flatten(), bl2], dim=0)
        return bl

    def _cal_angles(self, X, seg_id):
        ori_X = X
        X = X[:, :3].reshape(-1, 3)  # [N * 3, 3], N, CA, C
        U = F.normalize(X[1:] - X[:-1], dim=-1)  # [N * 3 - 1, 3]

        # 1. dihedral angles
        u_2, u_1, u_0 = U[:-2], U[1:-1], U[2:]  # [N * 3 - 3, 3]
        # backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)
        # angle between normals
        eps = 1e-7
        cosD = (n_2 * n_1).sum(-1)  # [(N-1) * 3]
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        # D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        seg_id_atom = seg_id.repeat(1, 3).flatten()  # [N * 3]
        same_chain_mask = sequential_and(
            seg_id_atom[:-3] == seg_id_atom[1:-2],
            seg_id_atom[1:-2] == seg_id_atom[2:-1],
            seg_id_atom[2:-1] == seg_id_atom[3:]
        )  # [N * 3 - 3]
        # D = D[same_chain_mask]
        cosD = cosD[same_chain_mask]

        # 2. bond angles (C_{n-1}-N, N-CA), (N-CA, CA-C), (CA-C, C=O), (CA-C, C-N_{n+1}), (O=C, C-Nn)
        u_0, u_1 = U[:-1], U[1:]  # [N*3 - 2, 3]
        cosA1 = ((-u_0) * u_1).sum(-1)  # [N*3 - 2], (C_{n-1}-N, N-CA), (N-CA, CA-C), (CA-C, C-N_{n+1})
        same_chain_mask = sequential_and(
            seg_id_atom[:-2] == seg_id_atom[1:-1],
            seg_id_atom[1:-1] == seg_id_atom[2:]
        )
        cosA1 = cosA1[same_chain_mask]  # [N*3 - 2 * num_chain]
        u_co = F.normalize(ori_X[:, 3] - ori_X[:, 2], dim=-1)  # [N, 3], C=O
        u_cca = -U[1::3]  # [N, 3], C-CA
        u_cn = U[2::3]  # [N-1, 3], C-N_{n+1}
        cosA2 = (u_co * u_cca).sum(-1)  # [N], (C=O, C-CA)
        cosA3 = (u_co[:-1] * u_cn).sum(-1)  # [N-1], (C=O, C-N_{n+1})
        same_chain_mask = (seg_id[:-1] == seg_id[1:])  # [N-1]
        cosA3 = cosA3[same_chain_mask]
        cosA = torch.cat([cosA1, cosA2, cosA3], dim=-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)

        return cosD, cosA

    def coord_loss(self, pred_X, true_X, batch_id, atom_mask, reference=None):
        pred_bb, true_bb = pred_X[:, :4], true_X[:, :4]
        bb_mask = atom_mask[:, :4]
        true_X = true_X.clone()
        ops = []

        align_obj = pred_bb if reference is None else reference[:, :4]

        for i in range(torch.max(batch_id) + 1):
            is_cur_graph = batch_id == i
            cur_bb_mask = bb_mask[is_cur_graph]
            _, R, t = kabsch_torch(
                true_bb[is_cur_graph][cur_bb_mask],
                align_obj[is_cur_graph][cur_bb_mask],
                requires_grad=True)
            true_X[is_cur_graph] = torch.matmul(true_X[is_cur_graph], R.T) + t
            ops.append((R.detach(), t.detach()))

        xloss = F.smooth_l1_loss(
            pred_X[atom_mask], true_X[atom_mask],
            reduction='sum') / atom_mask.sum()  # atom-level loss
        bb_rmsd = torch.sqrt(((pred_X[:, :4] - true_X[:, :4]) ** 2).sum(-1).mean(-1))  # [N]
        return xloss, bb_rmsd, ops

    def structure_loss(self, pred_X, true_X, S, res_idx, batch, full_profile=True):
        seg_id = self.aa_feature._construct_segment_ids(res_idx, batch)

        # loss of backbone (...N-CA-C(O)-N...) bond length
        true_bl = self._cal_backbone_bond_lengths(true_X, seg_id)
        pred_bl = self._cal_backbone_bond_lengths(pred_X, seg_id)
        bond_loss = F.smooth_l1_loss(pred_bl, true_bl)

        # loss of backbone dihedral angles
        if full_profile:
            true_cosD, true_cosA = self._cal_angles(true_X, seg_id)
            pred_cosD, pred_cosA = self._cal_angles(pred_X, seg_id)
            angle_loss = F.smooth_l1_loss(pred_cosD, true_cosD)
            bond_angle_loss = F.smooth_l1_loss(pred_cosA, true_cosA)

        # loss of sidechain bonds
        true_sc_bl = self._cal_sidechain_bond_lengths(S, true_X)
        pred_sc_bl = self._cal_sidechain_bond_lengths(S, pred_X)
        sc_bond_loss = F.smooth_l1_loss(pred_sc_bl, true_sc_bl)

        # loss of sidechain chis
        if full_profile:
            true_sc_chi = self._cal_sidechain_chis(S, true_X)
            pred_sc_chi = self._cal_sidechain_chis(S, pred_X)
            sc_chi_loss = F.smooth_l1_loss(pred_sc_chi, true_sc_chi)

        # exerting constraints on bond lengths only is sufficient
        loss = bond_loss + sc_bond_loss

        if full_profile:
            details = (loss, bond_loss, bond_angle_loss, angle_loss, sc_bond_loss, sc_chi_loss)
        else:
            details = (loss, bond_loss, sc_bond_loss)

        return loss


def sequential_and(*tensors):
    res = tensors[0]
    for mat in tensors[1:]:
        res = torch.logical_and(res, mat)
    return res
