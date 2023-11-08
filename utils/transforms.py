import sys
sys.path.append("..")
import copy
import os
import random
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from torch_geometric.transforms import Compose
from torch_geometric.nn.pool import knn_graph
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.data import Data, Batch
from torch_scatter import scatter_add
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem

from .data import ProteinLigandData
from .protein_ligand import ATOM_FAMILIES
from .chemutils import enumerate_assemble, list_filter, rand_rotate

# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(
            atom.GetAtomicNum())] + [allowable_features[
                                         'possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(
                bond.GetBondType())] + [allowable_features[
                'possible_bond_dirs'].index(
                bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data


class RefineData(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        # delete H atom of pocket
        protein_element = data.protein_element
        is_H_protein = (protein_element == 1)
        if torch.sum(is_H_protein) > 0:
            not_H_protein = ~is_H_protein
            data.protein_atom_name = list(compress(data.protein_atom_name, not_H_protein))
            data.protein_atom_to_aa_type = data.protein_atom_to_aa_type[not_H_protein]
            data.protein_element = data.protein_element[not_H_protein]
            data.protein_is_backbone = data.protein_is_backbone[not_H_protein]
            data.protein_pos = data.protein_pos[not_H_protein]
        # delete H atom of ligand
        ligand_element = data.ligand_element
        is_H_ligand = (ligand_element == 1)
        if torch.sum(is_H_ligand) > 0:
            not_H_ligand = ~is_H_ligand
            data.ligand_atom_feature = data.ligand_atom_feature[not_H_ligand]
            data.ligand_element = data.ligand_element[not_H_ligand]
            data.ligand_pos = data.ligand_pos[not_H_ligand]
            # nbh
            index_atom_H = torch.nonzero(is_H_ligand)[:, 0]
            index_changer = -np.ones(len(not_H_ligand), dtype=np.int64)
            index_changer[not_H_ligand] = np.arange(torch.sum(not_H_ligand))
            new_nbh_list = [value for ind_this, value in zip(not_H_ligand, data.ligand_nbh_list.values()) if ind_this]
            data.ligand_nbh_list = {i: [index_changer[node] for node in neigh if node not in index_atom_H] for i, neigh
                                    in enumerate(new_nbh_list)}
            # bond
            ind_bond_with_H = np.array([(bond_i in index_atom_H) | (bond_j in index_atom_H) for bond_i, bond_j in
                                        zip(*data.ligand_bond_index)])
            ind_bond_without_H = ~ind_bond_with_H
            old_ligand_bond_index = data.ligand_bond_index[:, ind_bond_without_H]
            data.ligand_bond_index = torch.tensor(index_changer)[old_ligand_bond_index]
            data.ligand_bond_type = data.ligand_bond_type[ind_bond_without_H]

        return data


class FocalBuilder(object):
    def __init__(self, close_threshold=0.8, max_bond_length=2.4):
        self.close_threshold = close_threshold
        self.max_bond_length = max_bond_length
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        # ligand_context_pos = data.ligand_context_pos
        # ligand_pos = data.ligand_pos
        ligand_masked_pos = data.ligand_masked_pos
        protein_pos = data.protein_pos
        context_idx = data.context_idx
        masked_idx = data.masked_idx
        old_bond_index = data.ligand_bond_index
        # old_bond_types = data.ligand_bond_type  # type: 0, 1, 2
        has_unmask_atoms = context_idx.nelement() > 0
        if has_unmask_atoms:
            # # get bridge bond index (mask-context bond)
            ind_edge_index_candidate = [
                (context_node in context_idx) and (mask_node in masked_idx)
                for mask_node, context_node in zip(*old_bond_index)
            ]  # the mask-context order is right
            bridge_bond_index = old_bond_index[:, ind_edge_index_candidate]
            # candidate_bond_types = old_bond_types[idx_edge_index_candidate]
            idx_generated_in_whole_ligand = bridge_bond_index[0]
            idx_focal_in_whole_ligand = bridge_bond_index[1]

            index_changer_masked = torch.zeros(masked_idx.max() + 1, dtype=torch.int64)
            index_changer_masked[masked_idx] = torch.arange(len(masked_idx))
            idx_generated_in_ligand_masked = index_changer_masked[idx_generated_in_whole_ligand]
            pos_generate = ligand_masked_pos[idx_generated_in_ligand_masked]

            data.idx_generated_in_ligand_masked = idx_generated_in_ligand_masked
            data.pos_generate = pos_generate

            index_changer_context = torch.zeros(context_idx.max() + 1, dtype=torch.int64)
            index_changer_context[context_idx] = torch.arange(len(context_idx))
            idx_focal_in_ligand_context = index_changer_context[idx_focal_in_whole_ligand]
            idx_focal_in_compose = idx_focal_in_ligand_context  # if ligand_context was not before protein in the compose, this was not correct
            data.idx_focal_in_compose = idx_focal_in_compose

            data.idx_protein_all_mask = torch.empty(0, dtype=torch.long)  # no use if has context
            data.y_protein_frontier = torch.empty(0, dtype=torch.bool)  # no use if has context

        else:  # # the initial atom. surface atoms between ligand and protein
            assign_index = radius(x=ligand_masked_pos, y=protein_pos, r=4., num_workers=16)
            if assign_index.size(1) == 0:
                dist = torch.norm(data.protein_pos.unsqueeze(1) - data.ligand_masked_pos.unsqueeze(0), p=2, dim=-1)
                assign_index = torch.nonzero(dist <= torch.min(dist) + 1e-5)[0:1].transpose(0, 1)
            idx_focal_in_protein = assign_index[0]
            data.idx_focal_in_compose = idx_focal_in_protein  # no ligand context, so all composes are protein atoms
            data.pos_generate = ligand_masked_pos[assign_index[1]]
            data.idx_generated_in_ligand_masked = torch.unique(assign_index[1])  # for real of the contractive transform

            data.idx_protein_all_mask = data.idx_protein_in_compose  # for input of initial frontier prediction
            y_protein_frontier = torch.zeros_like(data.idx_protein_all_mask,
                                                  dtype=torch.bool)  # for label of initial frontier prediction
            y_protein_frontier[torch.unique(idx_focal_in_protein)] = True
            data.y_protein_frontier = y_protein_frontier

        # generate not positions: around pos_focal ( with `max_bond_length` distance) but not close to true generated within `close_threshold`
        # pos_focal = ligand_context_pos[idx_focal_in_ligand_context]
        # pos_notgenerate = pos_focal + torch.randn_like(pos_focal) * self.max_bond_length  / 2.4
        # dist = torch.norm(pos_generate - pos_notgenerate, p=2, dim=-1)
        # ind_close = (dist < self.close_threshold)
        # while ind_close.any():
        #     new_pos_notgenerate = pos_focal[ind_close] + torch.randn_like(pos_focal[ind_close]) * self.max_bond_length  / 2.3
        #     dist[ind_close] = torch.norm(pos_generate[ind_close] - new_pos_notgenerate, p=2, dim=-1)
        #     pos_notgenerate[ind_close] = new_pos_notgenerate
        #     ind_close = (dist < self.close_threshold)
        # data.pos_notgenerate = pos_notgenerate

        return data


class AtomComposer(object):

    def __init__(self, protein_dim, ligand_dim, knn):
        super().__init__()
        self.protein_dim = protein_dim
        self.ligand_dim = ligand_dim
        self.knn = knn  # knn of compose atoms

    def __call__(self, data: ProteinLigandData):
        # fetch ligand context and protein from data
        ligand_context_pos = data['ligand_context_pos']
        ligand_context_feature_full = data['ligand_context_feature_full']
        protein_pos = data['protein_pos']
        protein_atom_feature = data['protein_atom_feature']
        len_ligand_ctx = len(ligand_context_pos)
        len_protein = len(protein_pos)

        # compose ligand context and protein. save idx of them in compose
        data['compose_pos'] = torch.cat([ligand_context_pos, protein_pos], dim=0)
        len_compose = len_ligand_ctx + len_protein
        ligand_context_feature_full_expand = torch.cat([
            ligand_context_feature_full,
            torch.zeros([len_ligand_ctx, self.protein_dim - self.ligand_dim], dtype=torch.long)
        ], dim=1)
        data['compose_feature'] = torch.cat([ligand_context_feature_full_expand, protein_atom_feature], dim=0)
        data['idx_ligand_ctx_in_compose'] = torch.arange(len_ligand_ctx, dtype=torch.long)  # can be delete
        data['idx_protein_in_compose'] = torch.arange(len_protein, dtype=torch.long) + len_ligand_ctx  # can be delete

        # build knn graph and bond type
        data = self.get_knn_graph(data, self.knn, len_ligand_ctx, len_compose, num_workers=16)
        return data

    @staticmethod
    def get_knn_graph(data: ProteinLigandData, knn, len_ligand_ctx, len_compose, num_workers=1, ):
        data['compose_knn_edge_index'] = knn_graph(data['compose_pos'], knn, flow='target_to_source', num_workers=num_workers)

        id_compose_edge = data['compose_knn_edge_index'][0,
                          :len_ligand_ctx * knn] * len_compose + data['compose_knn_edge_index'][1, :len_ligand_ctx * knn]
        id_ligand_ctx_edge = data['ligand_context_bond_index'][0] * len_compose + data['ligand_context_bond_index'][1]
        idx_edge = [torch.nonzero(id_compose_edge == id_) for id_ in id_ligand_ctx_edge]
        idx_edge = torch.tensor([a.squeeze() if len(a) > 0 else torch.tensor(-1) for a in idx_edge], dtype=torch.long)
        data['compose_knn_edge_type'] = torch.zeros(len(data['compose_knn_edge_index'][0]),
                                                 dtype=torch.long)  # for encoder edge embedding
        data['compose_knn_edge_type'][idx_edge[idx_edge >= 0]] = data['ligand_context_bond_type'][idx_edge >= 0]
        data['compose_knn_edge_feature'] = torch.cat([
            torch.ones([len(data['compose_knn_edge_index'][0]), 1], dtype=torch.long),
            torch.zeros([len(data['compose_knn_edge_index'][0]), 3], dtype=torch.long),
        ], dim=-1)
        data['compose_knn_edge_feature'][idx_edge[idx_edge >= 0]] = F.one_hot(data['ligand_context_bond_type'][idx_edge >= 0],
                                                                           num_classes=4)  # 0 (1,2,3)-onehot
        return data


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.atom_types = torch.arange(38)
        self.max_num_aa = 21

    @property
    def feature_dim(self):
        return 38

    def __call__(self, data: ProteinLigandData):
        atom_type = data['protein_atom_name'].view(-1, 1) == self.atom_types.view(1, -1)
        data['protein_atom_feature'] = atom_type.float()
        return data


class FeaturizeLigandAtom(object):

    def __init__(self):
        super().__init__()
        # self.atomic_numbers = torch.LongTensor([1,6,7,8,9,15,16,17])  # H C N O F P S Cl
        self.atomic_numbers = torch.LongTensor([6, 7, 8, 9, 15, 16, 17])  # C N O F P S Cl

    @property
    def num_properties(self):
        return len(ATOM_FAMILIES)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + len(ATOM_FAMILIES)

    def __call__(self, data: ProteinLigandData):
        element = data['ligand_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        x = torch.cat([element, data['ligand_atom_feature']], dim=-1)
        data['ligand_atom_feature'] = x.float()
        return data


class FeaturizeLigandBond(object):

    def __init__(self):
        super().__init__()

    def __call__(self, data: ProteinLigandData):
        data['ligand_bond_feature'] = F.one_hot(data['ligand_bond_type'] - 1, num_classes=3)  # (1,2,3) to (0,1,2)-onehot

        neighbor_dict = {}
        # used in rotation angle prediction
        mol = data['moltree'].mol
        for i, atom in enumerate(mol.GetAtoms()):
            neighbor_dict[i] = [n.GetIdx() for n in atom.GetNeighbors()]
        data['ligand_neighbors'] = neighbor_dict
        return data


class LigandCountNeighbors(object):

    @staticmethod
    def count_neighbors(edge_index, symmetry, valence=None, num_nodes=None):
        assert symmetry == True, 'Only support symmetrical edges.'

        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if valence is None:
            valence = torch.ones([edge_index.size(1)], device=edge_index.device)
        valence = valence.view(edge_index.size(1))

        return scatter_add(valence, index=edge_index[0], dim=0, dim_size=num_nodes).long()

    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data['ligand_num_neighbors'] = self.count_neighbors(
            data['ligand_bond_index'],
            symmetry=True,
            num_nodes=data['ligand_element'].size(0),
        )
        data['ligand_atom_valence'] = self.count_neighbors(
            data['ligand_bond_index'],
            symmetry=True,
            valence=data['ligand_bond_type'],
            num_nodes=data['ligand_element'].size(0),
        )
        return data


def kabsch(A, B):
    # Input:
    #     Nominal  A Nx3 matrix of points
    #     Measured B Nx3 matrix of points
    # Returns R,t
    # R = 3x3 rotation matrix (B to A)
    # t = 3x1 translation vector (B to A)
    assert len(A) == len(B)
    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    # center the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.transpose(BB) * AA
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T * U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T
    t = -R * centroid_B.T + centroid_A.T
    return R, t

