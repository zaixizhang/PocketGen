import copy
import sys
sys.path.append("..")
import numpy as np
import os
import torch
import torch.nn as nn
from torch.nn import Module, Linear, Embedding
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_mean
from torch_geometric.data import Data, Batch
from rdkit.Chem import ChemicalFeatures

from .encoders import get_encoder, MLP, CFTransformerEncoder
from .common import *

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

RES_ATOMS = [[ATOM_TYPES.index(i) for i in res if i != ''] for res in RES_ATOM14]


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
            residue_index = torch.arange(len(residue_mask_i))
            front = residue_index[~residue_mask_i][:2]
            end = residue_index[~residue_mask_i][-2:]
            near = nearest(residue_mask_i)
            for k in range(len(residue_mask_i)):
                if residue_mask_i[k]:
                    mask = atom2residue == (k + offset)
                    if k < front[0]:
                        pred_X[mask] = backbone_pos_i[front[0]] + (k - front[0]) / (front[0] - front[1]) * (backbone_pos_i[front[0]] - backbone_pos_i[front[1]])
                    elif k > end[1]:
                        pred_X[mask] = backbone_pos_i[end[1]] + (k - end[1]) / (end[1] - end[0]) * (backbone_pos_i[end[1]] - backbone_pos_i[end[0]])
                    else:
                        pred_X[mask] = ((k - near[k][0]) * backbone_pos_i[near[k][1]] + (near[k][1] - k) * backbone_pos_i[near[k][0]]) * 1 / (near[k][1] - near[k][0])
            offset += len(residue_mask_i)

    return pred_X


class Pocket_Design(Module):
    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim, device):
        super().__init__()
        self.config = config
        self.device = device
        self.protein_atom_emb = Linear(protein_atom_feature_dim, config.hidden_channels)
        self.ligand_atom_emb = Linear(ligand_atom_feature_dim, config.hidden_channels)
        self.encoder = get_encoder(config.encoder, device)  # hierachical graph transformer encoder
        self.residue_mlp = Linear(config.hidden_channels, 20)
        self.Softmax = nn.Softmax(dim=1)
        self.huber_loss = nn.SmoothL1Loss(reduction='mean')
        self.dist_loss = torch.nn.MSELoss(reduction='mean')
        self.pred_loss = nn.CrossEntropyLoss(reduction='mean')
        self.interpolate_steps = 5

    def compose_test(self, batch):
        h_ligand = self.ligand_atom_emb(batch['ligand_atom_feature'])
        h_protein = self.protein_atom_emb(batch['protein_atom_feature'])
        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=batch['protein_pos'],
                                                                         pos_ligand=batch['ligand_pos'],
                                                                         batch_protein=batch['protein_atom_batch'],
                                                                         batch_ligand=batch['ligand_atom_batch'])
        return h_ctx, pos_ctx, batch_ctx, mask_protein

    def compose(self, batch, pred_res_type, protein_pos, ligand_pos, backbone=False):
        if backbone:
            batch_protein = batch['protein_atom_batch_backbone']
            edit_atoms = batch['protein_edit_atom_backbone']
            protein_feature_mask = batch['protein_atom_feature_backbone'][edit_atoms]
            protein_feature_mask[:, -20:] = torch.repeat_interleave(pred_res_type.detach(),
                                                                    torch.ones(len(pred_res_type), dtype=int,
                                                                               device=self.device) * 4, dim=0)
            batch['protein_atom_feature_backbone'][edit_atoms] = protein_feature_mask
            h_protein = batch['protein_atom_feature_backbone']
        else:
            batch_protein = batch['protein_atom_batch']
            edit_atoms = batch['random_mask_atom']
            protein_feature_mask = batch['protein_atom_feature'][edit_atoms]
            protein_feature_mask[:, -20:] = torch.zeros(edit_atoms.sum(), 20, device=self.device)
            protein_feature_mask[:, -21] = 1
            batch['protein_atom_feature'][edit_atoms] = protein_feature_mask
            h_protein = batch['protein_atom_feature']

        h_ligand = self.ligand_atom_emb(batch['ligand_atom_feature'])
        h_protein = self.protein_atom_emb(h_protein)
        h_ctx, pos_ctx, batch_ctx, mask_protein = compose_context_stable(h_protein=h_protein, h_ligand=h_ligand,
                                                                         pos_protein=protein_pos,
                                                                         pos_ligand=ligand_pos,
                                                                         batch_protein=batch_protein,
                                                                         batch_ligand=batch['ligand_atom_batch'])
        return h_ctx, pos_ctx, batch_ctx, mask_protein

    def forward(self, batch):
        external_index = compose_external_attention(copy.deepcopy(batch['protein_atom_batch_backbone']),
                                                    copy.deepcopy(batch['ligand_atom_batch']),
                                                    copy.deepcopy(batch['protein_edit_atom_backbone']))
        external_index1 = compose_external_attention(copy.deepcopy(batch['protein_atom_batch']),
                                                     copy.deepcopy(batch['ligand_atom_batch']),
                                                     copy.deepcopy(batch['protein_edit_atom']))
        residue_mask = batch['protein_edit_residue']
        self.pred_res_type = torch.ones(residue_mask.sum(), 20, device=self.device) / 20
        # backbone
        loss_list = [0., 0.]
        for t in range(self.interpolate_steps):
            atom_mask = batch['protein_edit_atom_backbone']
            # Interpolated label
            ratio = (self.interpolate_steps - t) / self.interpolate_steps
            label_X, pred_X = copy.deepcopy(batch['protein_pos_backbone']), copy.deepcopy(batch['protein_pos_backbone'])
            label_ligand, pred_ligand = copy.deepcopy(batch['ligand_pos']), copy.deepcopy(batch['ligand_pos'])
            pred_ligand += torch.randn_like(pred_ligand).to(self.device) * ratio * 0.5
            pred_X = interpolation_init(pred_X, residue_mask, copy.deepcopy(batch['residue_pos']), batch['atom2residue_backbone'],
                                        batch['protein_atom_batch_backbone'], batch['amino_acid_batch'])
            pred_X = (1 - ratio) * label_X + ratio * pred_X

            h_ctx, pos_ctx, batch_ctx, mask_protein = self.compose(copy.deepcopy(batch), self.pred_res_type.detach(), pred_X, pred_ligand, backbone=True)
            h_ctx, h_residue, pred_X, pred_ligand = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch_ctx=batch_ctx,
                                                                 batch=copy.deepcopy(batch), mask_protein=mask_protein,
                                                                 pred_res_type=self.pred_res_type.detach(),
                                                                 external_index=external_index, backbone=True)
            loss_list[0] += self.huber_loss(pred_X[atom_mask], label_X[atom_mask]) + self.huber_loss(pred_ligand, label_ligand)
            self.pred_res_type = self.residue_mlp(h_residue[residue_mask])
            loss_list[1] += self.pred_loss(self.pred_res_type, batch['amino_acid'][residue_mask] - 1)

        # full atom
        random_mask = batch['random_mask_residue']
        for t in range(self.interpolate_steps):
            atom_mask = batch['protein_edit_atom']
            # Interpolated label
            ratio = (self.interpolate_steps - t) / self.interpolate_steps
            label_X, pred_X = copy.deepcopy(batch['protein_pos']), copy.deepcopy(batch['protein_pos'])
            label_ligand, pred_ligand = copy.deepcopy(batch['ligand_pos']), copy.deepcopy(batch['ligand_pos'])
            pred_ligand += torch.randn_like(pred_ligand).to(self.device) * ratio * 0.2
            pred_X[atom_mask] += torch.randn_like(batch['protein_pos'][atom_mask]).to(self.device) * ratio * 0.3
            index = torch.arange(len(batch['amino_acid']))[batch['protein_edit_residue']]
            for k in range(len(batch['amino_acid'])):
                mask = batch['atom2residue'] == k
                if k in index:
                    pos = pred_X[mask]
                    pos[4:] = ratio * (pos[1].repeat(len(pos) - 4, 1) + 0.1 * torch.randn(len(pos) - 4, 3, device=self.device)) + (1 - ratio) * pos[4:]
                    pred_X[mask] = pos

            h_ctx, pos_ctx, batch_ctx, mask_protein = self.compose(copy.deepcopy(batch), self.pred_res_type.detach(), pred_X, pred_ligand, backbone=False)
            h_ctx, h_residue, pred_X, pred_ligand = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch_ctx=batch_ctx,
                                                        batch=copy.deepcopy(batch), mask_protein=mask_protein,
                                                        pred_res_type=self.pred_res_type.detach(),
                                                        external_index=external_index1, backbone=False)
            loss_list[0] += self.huber_loss(pred_X[atom_mask], label_X[atom_mask]) + self.huber_loss(pred_ligand, label_ligand)
            loss_list[1] += self.pred_loss(self.residue_mlp(h_residue[random_mask]), batch['amino_acid'][random_mask] - 1)

        return loss_list[1] + loss_list[0], loss_list

    def generate(self, batch):
        print('Start Generating')
        external_index = compose_external_attention(copy.deepcopy(batch['protein_atom_batch_backbone']),
                                                    copy.deepcopy(batch['ligand_atom_batch']),
                                                    copy.deepcopy(batch['protein_edit_atom_backbone']))
        residue_mask = batch['protein_edit_residue']
        self.pred_res_type = torch.ones(residue_mask.sum(), 20, device=self.device) / 20
        # backbone
        label_S = copy.deepcopy(batch['amino_acid'])
        atom_mask = batch['protein_edit_atom_backbone']
        label_X, pred_X = copy.deepcopy(batch['protein_pos_backbone']), batch['protein_pos_backbone']
        pred_X = interpolation_init(pred_X, residue_mask, batch['residue_pos'], batch['atom2residue_backbone'],
                                    batch['protein_atom_batch_backbone'], batch['amino_acid_batch'])
        for t in range(5):
            h_ctx, pos_ctx, batch_ctx, mask_protein = self.compose(copy.deepcopy(batch), self.pred_res_type, pred_X,
                                                                   batch['ligand_pos'], backbone=True)
            h_ctx, h_residue, pred_X, batch['ligand_pos'] = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch_ctx=batch_ctx,
                                                            batch=copy.deepcopy(batch), mask_protein=mask_protein,
                                                            pred_res_type=self.pred_res_type,
                                                            external_index=external_index, backbone=True)
            self.pred_res_type = self.residue_mlp(h_residue[residue_mask])

        batch['protein_pos'][batch['edit_backbone']] = pred_X[atom_mask]
        select = torch.argmax(self.Softmax(self.pred_res_type), 1).view(-1)
        self.pred_res_type = torch.zeros_like(self.pred_res_type, device=self.device)
        self.pred_res_type[torch.arange(len(select)), select] = 1

        # full atom
        batch['amino_acid'][batch['protein_edit_residue']] = select + 1
        batch['random_mask_residue'] = batch['protein_edit_residue']
        batch = random_mask(batch, device=self.device, mask=False)
        for t in range(1, 10):  # self-consistent iterative steps
            external_index1 = compose_external_attention(copy.deepcopy(batch['protein_atom_batch']),
                                                         copy.deepcopy(batch['ligand_atom_batch']),
                                                         copy.deepcopy(batch['protein_edit_atom']))
            for s in range(5):  # refinement steps
                h_ctx, pos_ctx, batch_ctx, mask_protein = self.compose_test(batch)
                h_ctx, h_residue, batch['protein_pos'], batch['ligand_pos'] = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch_ctx=batch_ctx,
                                                                      batch=copy.deepcopy(batch),
                                                                      mask_protein=mask_protein,
                                                                      pred_res_type=self.pred_res_type.detach(),
                                                                      external_index=external_index1, backbone=False,
                                                                      mask=False)
            # random mask
            if t == 10:
                continue
            batch = random_mask(batch, device=self.device, mask=True)
            external_index1 = compose_external_attention(copy.deepcopy(batch['protein_atom_batch']),
                                                         copy.deepcopy(batch['ligand_atom_batch']),
                                                         copy.deepcopy(batch['protein_edit_atom']))
            h_ctx, pos_ctx, batch_ctx, mask_protein = self.compose(batch, self.pred_res_type.detach(),
                                                                   batch['protein_pos'], batch['ligand_pos'],
                                                                   backbone=False)
            h_ctx, h_residue, batch['protein_pos'], batch['ligand_pos'] = self.encoder(node_attr=h_ctx, pos=pos_ctx, batch_ctx=batch_ctx,
                                                                  batch=copy.deepcopy(batch), mask_protein=mask_protein,
                                                                  pred_res_type=self.pred_res_type.detach(),
                                                                  external_index=external_index1, backbone=False)
            batch['amino_acid'][batch['random_mask_residue']] = torch.argmax(self.residue_mlp(h_residue[batch['random_mask_residue']]), dim=1) + 1
            self.pred_res_type = torch.zeros_like(self.pred_res_type, device=self.device)
            self.pred_res_type[torch.arange(len(self.pred_res_type)), batch['amino_acid'][residue_mask] - 1] = 1
            batch = random_mask(batch, device=self.device, mask=False)

        aar = (label_S[residue_mask] == batch['amino_acid'][residue_mask]).sum() / len(label_S[residue_mask])
        rmsd = torch.sqrt((label_X[atom_mask] - batch['protein_pos'][batch['edit_backbone']]).norm(dim=1).sum() / atom_mask.sum())
        return aar, rmsd


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
        batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(batch['residue_natoms']), device=device), batch['residue_natoms'])
        batch['protein_edit_atom'] = torch.repeat_interleave(batch['protein_edit_residue'], batch['residue_natoms'], dim=0)
        batch['random_mask_atom'] = torch.repeat_interleave(batch['random_mask_residue'], batch['residue_natoms'], dim=0)
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
                        batch['protein_pos'][mask][1:2].repeat(sidechain_size, 1) + 0.1 * torch.randn(sidechain_size, 3, device=device))
                feature_tmp.append(atom_feature(res_type, device))
                natoms_tmp.append(NUM_ATOMS[res_type])
            else:
                pos_tmp.append(batch['protein_pos'][mask])
                feature_tmp.append(batch['protein_atom_feature'][mask])
                natoms_tmp.append(batch['protein_pos'][mask].shape[0])
        batch['protein_pos'], batch['protein_atom_feature'] = torch.cat(pos_tmp, dim=0), torch.cat(feature_tmp, dim=0)
        batch['protein_atom_feature'][:, -21] = 0

        batch['residue_natoms'] = torch.tensor(natoms_tmp, device=device)
        batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(batch['residue_natoms']), device=device), batch['residue_natoms'])
        batch['protein_edit_atom'] = torch.repeat_interleave(batch['protein_edit_residue'], batch['residue_natoms'], dim=0)

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


def write_pdb(protein_pos, amino_acid, atom_name, res_batch):
    lines = ['HEADER    POCKET', 'COMPND    POCKET']
    path = '/data/saved'
    with open('', 'w') as f:
        f.writelines(lines)
        for i in range(len(protein_pos)):
            j0 = str('ATOM').ljust(6)  # atom#6s
            j1 = str(i).rjust(5)  # aomnum#5d
            j2 = str(atom_name[i]).center(4)  # atomname$#4s
            j3 = amino_acid[i].ljust(3)  # resname#1s
            j4 = str('A').rjust(1)  # Astring
            j5 = str(res_batch[i]).rjust(4)  # resnum
            j6 = str('%8.3f' % (float(protein_pos[i][0]))).rjust(8)  # x
            j7 = str('%8.3f' % (float(protein_pos[i][1]))).rjust(8)  # y
            j8 = str('%8.3f' % (float(protein_pos[i][2]))).rjust(8)  # z\
            j9 = str('%6.2f' % (1.00)).rjust(6)  # occ
            j10 = str('%6.2f' % (25.02)).ljust(6)  # temp
            j11 = str(atom_name[i][0]).rjust(12)  # elname
            f.write("%s%s %s %s %s%s    %s%s%s%s%s%s\n" % j0, j1, j2, j3, j4, j5, j6, j7, j8, j9, j10, j11)
        f.write('END')
        f.write('\n')
