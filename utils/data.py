import copy
import random
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_sum
# from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset

FOLLOW_BATCH = ['protein_element', 'ligand_context_element', 'pos_real', 'pos_fake']


class ProteinLigandData(object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        # instance['ligand_nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) if instance.ligand_bond_index[0, k].item() == i] for i in instance.ligand_bond_index[0]}
        return instance


def batch_from_data_list(data_list):
    return Batch.from_data_list(data_list, follow_batch=['ligand_element', 'protein_element'])


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output


def collate_mols(mol_dicts):
    data_batch = {}
    batch_size = len(mol_dicts)
    for key in ['protein_pos', 'protein_atom_feature', 'ligand_pos', 'ligand_atom_feature',
                'protein_edit_residue', 'amino_acid', 'res_idx', 'residue_natoms', 'protein_atom_to_aa_type']:
        data_batch[key] = torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0)
    # residue pos
    data_batch['residue_pos'] = \
        torch.cat([torch.cat([mol_dict[key] for mol_dict in mol_dicts], dim=0).unsqueeze(0) for key in ['pos_N', 'pos_CA', 'pos_C', 'pos_O']], dim=0).permute(1,0,2)

    # random mask residues for the second stage (one residue per protein)
    tmp = []
    for mol_dict in mol_dicts:
        ind = torch.multinomial(mol_dict['protein_edit_residue'].float(), 1)
        selected = torch.zeros_like(mol_dict['protein_edit_residue'], dtype=bool)
        selected[ind] = 1
        tmp.append(selected)
    data_batch['random_mask_residue'] = torch.cat(tmp, dim=0)

    # remove side chains for the masked atoms
    num_residues = len(data_batch['amino_acid'])
    data_batch['atom2residue'] = torch.repeat_interleave(torch.arange(num_residues), data_batch['residue_natoms'])
    index1 = torch.arange(len(data_batch['amino_acid']))[data_batch['random_mask_residue']]
    index2 = torch.arange(len(data_batch['amino_acid']))[data_batch['protein_edit_residue']]
    for key in ['protein_pos', 'protein_atom_feature']:
        tmp1, tmp2 = [], []
        for k in range(num_residues):
            mask = data_batch['atom2residue'] == k
            if k in index1:
                tmp1.append(data_batch[key][mask][:4])
            else:
                tmp1.append(data_batch[key][mask])
            if k in index2:
                tmp2.append(data_batch[key][mask][:4])
            else:
                tmp2.append(data_batch[key][mask])
        data_batch[key] = torch.cat(tmp1, dim=0)
        data_batch[key + '_backbone'] = torch.cat(tmp2, dim=0)

    data_batch['residue_natoms'][data_batch['random_mask_residue']] = 4
    data_batch['atom2residue'] = torch.repeat_interleave(torch.arange(len(data_batch['residue_natoms'])), data_batch['residue_natoms'])
    # follow batch
    for key in ['ligand_atom_feature', 'amino_acid']:
        repeats = torch.tensor([len(mol_dict[key]) for mol_dict in mol_dicts])
        if key == 'amino_acid':
            data_batch['amino_acid_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)
        else:
            data_batch['ligand_atom_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)
    repeats = scatter_sum(data_batch['residue_natoms'], data_batch['amino_acid_batch'], dim=0)
    data_batch['protein_atom_batch'] = torch.repeat_interleave(torch.arange(batch_size), repeats)

    # backbone protein for the first stage

    data_batch['residue_natoms_backbone'] = copy.deepcopy(data_batch['residue_natoms'])
    data_batch['residue_natoms_backbone'][data_batch['protein_edit_residue']] = 4

    repeats = scatter_sum(data_batch['residue_natoms_backbone'], data_batch['amino_acid_batch'], dim=0)
    data_batch['protein_atom_batch_backbone'] = torch.repeat_interleave(torch.arange(batch_size), repeats)
    data_batch['atom2residue_backbone'] = torch.repeat_interleave(torch.arange(len(data_batch['residue_natoms_backbone'])), data_batch['residue_natoms_backbone'])
    data_batch['protein_edit_atom'] = torch.repeat_interleave(data_batch['protein_edit_residue'], data_batch['residue_natoms'], dim=0)
    data_batch['protein_edit_atom_backbone'] = torch.repeat_interleave(data_batch['protein_edit_residue'], data_batch['residue_natoms_backbone'], dim=0)
    data_batch['random_mask_atom'] = torch.repeat_interleave(data_batch['random_mask_residue'], data_batch['residue_natoms'], dim=0)

    data_batch['edit_sidechain'] = copy.deepcopy(data_batch['protein_edit_atom'])
    data_batch['edit_backbone'] = copy.deepcopy(data_batch['protein_edit_atom'])
    index = torch.arange(len(data_batch['amino_acid']))[data_batch['protein_edit_residue']]
    for k in range(num_residues):
        mask = data_batch['atom2residue'] == k
        if k in index:
            data_mask1, data_mask2 = data_batch['edit_sidechain'][mask], data_batch['edit_backbone'][mask]
            data_mask1[:4], data_mask2[4:] = 0, 0
            data_batch['edit_sidechain'][mask] = data_mask1
            data_batch['edit_backbone'][mask] = data_mask2
    return data_batch


