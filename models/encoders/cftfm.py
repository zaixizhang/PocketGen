CUDA_LAUNCH_BLOCKING = 1
import sys
sys.path.append("..")
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Sequential, ModuleList, Linear, Conv1d
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import add_self_loops
from torch_scatter import scatter_sum, scatter_softmax, scatter_mean, scatter_std
import numpy as np
from .radial_basis import RadialBasis
import copy
from math import pi as PI

from ..common import GaussianSmearing, ShiftedSoftplus
from ..protein_features import ProteinFeatures


residue_atom_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]]).float()

class AttentionInteractionBlock(Module):

    def __init__(self, hidden_channels, edge_channels, key_channels, num_heads=1):
        super().__init__()

        assert hidden_channels % num_heads == 0
        assert key_channels % num_heads == 0

        self.hidden_channels = hidden_channels
        self.key_channels = key_channels
        self.num_heads = num_heads

        self.k_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.q_lin = Conv1d(hidden_channels, key_channels, 1, groups=num_heads, bias=False)
        self.v_lin = Conv1d(hidden_channels, hidden_channels, 1, groups=num_heads, bias=False)

        self.weight_k_net = Sequential(
            Linear(edge_channels, key_channels // num_heads),
            ShiftedSoftplus(),
            Linear(key_channels // num_heads, key_channels // num_heads),
        )
        self.weight_k_lin = Linear(key_channels // num_heads, key_channels // num_heads)

        self.weight_v_net = Sequential(
            Linear(edge_channels, hidden_channels // num_heads),
            ShiftedSoftplus(),
            Linear(hidden_channels // num_heads, hidden_channels // num_heads),
        )
        self.weight_v_lin = Linear(hidden_channels // num_heads, hidden_channels // num_heads)

        self.centroid_lin = Linear(hidden_channels, hidden_channels)
        self.act = ShiftedSoftplus()
        self.out_transform = Linear(hidden_channels, hidden_channels)
        self.layernorm_attention = nn.LayerNorm(hidden_channels)
        self.layernorm_ffn = nn.LayerNorm(hidden_channels)

    def forward(self, x, edge_index, edge_attr):
        """
        Args:
            x:  Node features, (N, H).
            edge_index: (2, E).
            edge_attr:  (E, H)
        """
        N = x.size(0)
        row, col = edge_index  # (E,) , (E,)

        # self-attention layer_norm
        y = self.layernorm_attention(x)

        # Project to multiple key, query and value spaces
        h_keys = self.k_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1)  # (N, heads, K_per_head)
        h_queries = self.q_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1)  # (N, heads, K_per_head)
        h_values = self.v_lin(y.unsqueeze(-1)).view(N, self.num_heads, -1)  # (N, heads, H_per_head)

        # Compute keys and queries
        W_k = self.weight_k_net(edge_attr)  # (E, K_per_head)
        keys_j = self.weight_k_lin(W_k.unsqueeze(1) * h_keys[col])  # (E, heads, K_per_head)
        queries_i = h_queries[row]  # (E, heads, K_per_head)

        # Compute attention weights (alphas)
        qk_ij = (queries_i * keys_j).sum(-1)  # (E, heads)
        alpha = scatter_softmax(qk_ij, row, dim=0)

        # Compose messages
        W_v = self.weight_v_net(edge_attr)  # (E, H_per_head)
        msg_j = self.weight_v_lin(W_v.unsqueeze(1) * h_values[col])  # (E, heads, H_per_head)
        msg_j = alpha.unsqueeze(-1) * msg_j  # (E, heads, H_per_head)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=N).view(N, -1)  # (N, heads*H_per_head)
        x = aggr_msg + x
        y = self.layernorm_ffn(x)
        out = self.out_transform(self.act(y)) + x
        return out


class CFTransformerEncoder(Module):

    def __init__(self, hidden_channels=128, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=32,
                 cutoff=10.0):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlock(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                key_channels=key_channels,
                num_heads=num_heads,
            )
            self.interactions.append(block)

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch):
        # edge_index = radius_graph(pos, self.cutoff, batch=batch, loop=False)
        edge_index = knn_graph(pos, k=self.k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_attr)
        return h


# residue level graph transformer
class AAEmbedding(nn.Module):

    def __init__(self, device):
        super(AAEmbedding, self).__init__()

        self.hydropathy = {'#': 0, "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, "A": 1.8, "W": -0.9,
                           "G": -0.4, "T": -0.7, "S": -0.8, "Y": -1.3, "P": -1.6, "H": -3.2, "N": -3.5, "D": -3.5,
                           "Q": -3.5, "E": -3.5, "K": -3.9, "R": -4.5}
        self.volume = {'#': 0, "G": 60.1, "A": 88.6, "S": 89.0, "C": 108.5, "D": 111.1, "P": 112.7, "N": 114.1,
                       "T": 116.1, "E": 138.4, "V": 140.0, "Q": 143.8, "H": 153.2, "M": 162.9, "I": 166.7, "L": 166.7,
                       "K": 168.6, "R": 173.4, "F": 189.9, "Y": 193.6, "W": 227.8}
        self.charge = {**{'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}, **{x: 0 for x in 'ABCFGIJLMNOPQSTUVWXYZ#'}}
        self.polarity = {**{x: 1 for x in 'RNDQEHKSTY'}, **{x: 0 for x in "ACGILMFPWV#"}}
        self.acceptor = {**{x: 1 for x in 'DENQHSTY'}, **{x: 0 for x in "RKWACGILMFPV#"}}
        self.donor = {**{x: 1 for x in 'RKWNQHSTY'}, **{x: 0 for x in "DEACGILMFPV#"}}
        ALPHABET = ['#', 'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y',
                    'V']
        self.embedding = torch.tensor([
            [self.hydropathy[aa], self.volume[aa] / 100, self.charge[aa], self.polarity[aa], self.acceptor[aa],
             self.donor[aa]]
            for aa in ALPHABET]).to(device)

    def to_rbf(self, D, D_min, D_max, stride):
        D_count = int((D_max - D_min) / stride)
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view(1, -1)  # [1, K]
        D_expand = torch.unsqueeze(D, -1)  # [N, 1]
        return torch.exp(-((D_expand - D_mu) / stride) ** 2)

    def transform(self, aa_vecs):
        return torch.cat([
            self.to_rbf(aa_vecs[:, 0], -4.5, 4.5, 0.1),
            self.to_rbf(aa_vecs[:, 1], 0, 2.2, 0.1),
            self.to_rbf(aa_vecs[:, 2], -1.0, 1.0, 0.25),
            torch.sigmoid(aa_vecs[:, 3:] * 6 - 3),
        ], dim=-1)

    def dim(self):
        return 90 + 22 + 8 + 3

    def forward(self, x, raw=False):
        # B, N = x.size(0), x.size(1)
        # aa_vecs = self.embedding[x.view(-1)].view(B, N, -1)
        aa_vecs = self.embedding[x.view(-1)]
        rbf_vecs = self.transform(aa_vecs)
        return aa_vecs if raw else rbf_vecs

    def soft_forward(self, x):
        aa_vecs = torch.matmul(x, self.embedding)
        rbf_vecs = self.transform(aa_vecs)
        return rbf_vecs


class TransformerLayer(nn.Module):
    def __init__(self, num_hidden, num_heads=4, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.dropout_attention = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)
        self.self_attention_norm = nn.LayerNorm(num_hidden)
        self.ffn_norm = nn.LayerNorm(num_hidden)

        self.attention = ResidueAttention(num_hidden, num_heads)
        self.ffn = PositionWiseFeedForward(num_hidden, num_hidden)

    def forward(self, h_V, h_E, E_idx):
        """ Parallel computation of full transformer layer """
        # Self-attention
        y = self.self_attention_norm(h_V)
        y = self.attention(y, h_E, E_idx)
        h_V = h_V + self.dropout_attention(y)

        # Position-wise feedforward
        y = self.ffn_norm(h_V)
        y = self.ffn(y)
        h_V = h_V + self.dropout_ffn(y)
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class ResidueAttention(nn.Module):
    def __init__(self, num_hidden, num_heads=4):
        super(ResidueAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_hidden * 2, num_hidden, bias=False)
        self.W_V = nn.Linear(num_hidden * 2, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)
        self.act = ShiftedSoftplus()
        self.layernorm = nn.LayerNorm(num_hidden)

    def forward(self, h_V, h_E, edge_index):
        """ Self-attention, graph-structured O(Nk)
        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_hidden]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V:            Node update
        """

        # Queries, Keys, Values
        n_edges = h_E.shape[0]
        n_nodes = h_V.shape[0]
        n_heads = self.num_heads
        row, col = edge_index  # (E,) , (E,)

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_nodes, n_heads, 1, d])
        K = self.W_K(torch.cat([h_E, h_V[col]], dim=-1)).view([n_edges, n_heads, d, 1])
        V = self.W_V(torch.cat([h_E, h_V[col]], dim=-1)).view([n_edges, n_heads, d])
        # Attention with scaled inner product
        attend_logits = torch.matmul(Q[row], K).view([n_edges, n_heads])  # (E, heads)
        alpha = scatter_softmax(attend_logits, row, dim=0) / np.sqrt(d)
        # Compose messages
        msg_j = alpha.unsqueeze(-1) * V  # (E, heads, H_per_head)

        # Aggregate messages
        aggr_msg = scatter_sum(msg_j, row, dim=0, dim_size=n_nodes).view(n_nodes, -1)  # (N, heads*H_per_head)
        h_V_update = self.W_O(self.act(aggr_msg))
        return h_V_update


# hierachical graph transformer encoder
class HierEncoder(Module):
    def __init__(self, hidden_channels=128, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=32,
                 cutoff=10.0, device='cuda:0'):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff
        self.device = device

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)
        self.interactions = ModuleList()
        for _ in range(num_interactions):
            block = AttentionInteractionBlock(
                hidden_channels=hidden_channels,
                edge_channels=edge_channels,
                key_channels=key_channels,
                num_heads=num_heads,
            )
            self.interactions.append(block)

        # Residue level settings
        self.residue_feat = AAEmbedding(device)  # for residue node feature
        self.features = ProteinFeatures(top_k=8)  # for residue edge feature
        self.W_v = nn.Linear(hidden_channels + self.residue_feat.dim(), hidden_channels, bias=True)
        self.W_e = nn.Linear(self.features.feature_dimensions, hidden_channels, bias=True)
        self.residue_encoder_layers = nn.ModuleList([TransformerLayer(hidden_channels, dropout=0.1) for _ in range(2)])

        self.T_a = nn.Sequential(nn.Linear(2 * hidden_channels + edge_channels, hidden_channels), nn.ReLU(),
                                 nn.Linear(hidden_channels, 1))
        self.T_x = nn.Sequential(nn.Linear(3 * hidden_channels, hidden_channels), nn.ReLU(),
                                 nn.Linear(hidden_channels, 14))

    @property
    def out_channels(self):
        return self.hidden_channels

    def forward(self, node_attr, pos, batch_ctx, batch, pred_res_type, mask_protein, external_index, backbone=True,
                mask=True):
        S_id, R = batch['res_idx'], batch['amino_acid']
        residue_batch, atom2residue = batch['amino_acid_batch'], batch['atom2residue']
        edit_residue, edit_atom = batch['protein_edit_residue'], batch['protein_edit_atom']
        if mask:
            R[batch['random_mask_residue']] = 0
        R = F.one_hot(R, num_classes=21).float()
        if backbone:
            atom2residue, edit_atom = batch['atom2residue_backbone'], batch['protein_edit_atom_backbone']
            R_edit = R[edit_residue]
            R_edit[:, 1:] = pred_res_type
            R[edit_residue] = R_edit
            R[:, 0] = 0

        edge_index = knn_graph(pos, k=self.k, batch=batch_ctx, flow='target_to_source')
        edge_length = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1)
        edge_attr = self.distance_expansion(edge_length)

        h = node_attr
        for interaction in self.interactions:
            h = interaction(h, edge_index, edge_attr)

        h_ligand_coarse = scatter_sum(h[~mask_protein], batch['ligand_atom_batch'], dim=0)
        pos_ligand_coarse = scatter_sum(batch['ligand_pos'], batch['ligand_atom_batch'], dim=0)
        E, residue_edge_index, residue_edge_length, edge_index_new, E_new = self.features(pos_ligand_coarse,
                                                                                          batch['protein_edit_residue'],
                                                                                          batch['residue_pos'], S_id,
                                                                                          residue_batch)
        h_protein = h[mask_protein]
        V = torch.cat([self.residue_feat.soft_forward(R), scatter_sum(h_protein, atom2residue, dim=0)], dim=-1)
        h_res = self.W_v(V)

        h_res = torch.cat([h_res, h_ligand_coarse])
        edge_index_combined = torch.cat([residue_edge_index, edge_index_new], 1)
        E = torch.cat([E, E_new], 0)
        h_E = self.W_e(E)

        for layer in self.residue_encoder_layers:
            h_res = layer(h_res, h_E, edge_index_combined)

        # update X:
        h_res = h_res[:len(residue_batch)]
        h_E = h_E[:residue_edge_index.size(1)]
        # protein internal update
        mij = torch.cat([h_res[residue_edge_index[0]], h_res[residue_edge_index[1]], h_E], dim=-1)
        if backbone:
            protein_pos = pos[mask_protein]
            ligand_pos = pos[~mask_protein]
            N = atom2residue.max() + 1
            X_bb = torch.zeros(N, 4, 3).to(pos.device)
            for j in range(N):
                X_bb[j] = protein_pos[atom2residue == j][:4]  # 4 backbone atoms [N,4,3]
            xij = X_bb[residue_edge_index[0]] - X_bb[residue_edge_index[1]]  # [N,4,3]
            dij = xij.norm(dim=-1) + 1e-6  # [N,4]
            fij = torch.maximum(self.T_x(mij)[:, :4], 3.8 - dij)  # break term [N,4]
            xij = xij / dij.unsqueeze(-1) * fij.unsqueeze(-1)
            f_res = scatter_mean(xij, residue_edge_index[0], dim=0)  # [N,4,3]
            X_bb[edit_residue] += f_res.clamp(min=-5.0, max=5.0)[edit_residue]

            # Clash correction
            for _ in range(2):
                xij = X_bb[residue_edge_index[0]] - X_bb[residue_edge_index[1]]  # [N,4,3]
                dij = xij.norm(dim=-1) + 1e-6  # [N,4]
                fij = F.relu(3.8 - dij)  # repulsion term [N,4]
                xij = xij / dij.unsqueeze(-1) * fij.unsqueeze(-1)
                f_res = scatter_mean(xij, residue_edge_index[0], dim=0)  # [N,4,3]
                X_bb[edit_residue] += f_res.clamp(min=-5.0, max=5.0)[edit_residue]

            # protein-ligand external update
            protein_pos[edit_atom] = X_bb[edit_residue].view(-1, 3)
            pos[mask_protein] = protein_pos
            dij = torch.norm(pos[mask_protein][external_index[0]] - pos[~mask_protein][external_index[1]], dim=1) + 1e-6
            mij = torch.cat(
                [h[mask_protein][external_index[0]], h[~mask_protein][external_index[1]], self.distance_expansion(dij)],
                dim=-1)
            xij = pos[mask_protein][external_index[0]] - pos[~mask_protein][external_index[1]]
            fij = torch.maximum(self.T_a(mij).squeeze(-1), 1.5 - dij)
            xij = xij / dij.unsqueeze(-1) * fij.unsqueeze(-1)
            f_atom = scatter_mean(xij, external_index[0], dim=0, dim_size=protein_pos.size(0))
            protein_pos += f_atom
            f_ligand_atom = scatter_mean(xij, external_index[1], dim=0, dim_size=ligand_pos.size(0))
            ligand_pos -= f_ligand_atom * 0.05

        else:
            protein_pos = pos[mask_protein]
            ligand_pos = pos[~mask_protein]
            X_avg = scatter_mean(protein_pos, atom2residue, dim=0)
            X = X_avg.unsqueeze(1).repeat(1, 14, 1)
            N = atom2residue.max() + 1
            mask = torch.zeros(N, 14, dtype=bool).to(protein_pos.device)
            residue_natoms = atom2residue.bincount()
            for j in range(N):
                mask[j][:residue_natoms[j]] = 1  # all atoms mask [N,14]
                X[j][:residue_natoms[j]] = protein_pos[atom2residue == j]

            xij = X[residue_edge_index[0]] - X_avg[residue_edge_index[1]].unsqueeze(1)  # [N,14,3]
            dij = xij.norm(dim=-1) + 1e-6  # [N,14]
            fij = torch.maximum(self.T_x(mij), 3.8 - dij)  # break term [N,14]
            xij = xij / dij.unsqueeze(-1) * fij.unsqueeze(-1)
            f_res = scatter_mean(xij, residue_edge_index[0], dim=0)  # [N,14,3]
            f_res[:, :4] *= 0.1
            X[edit_residue] += f_res.clamp(min=-5.0, max=5.0)[edit_residue]

            for _ in range(2):
                protein_pos = X[mask]
                X_avg = scatter_mean(protein_pos, atom2residue, dim=0)
                xij = X[residue_edge_index[0]] - X_avg[residue_edge_index[1]].unsqueeze(1)  # [N,14,3]
                dij = xij.norm(dim=-1) + 1e-6  # [N,14]
                fij = F.relu(3.8 - dij)  # repulsion term [N,14]
                xij = xij / dij.unsqueeze(-1) * fij.unsqueeze(-1)
                f_res = scatter_mean(xij, residue_edge_index[0], dim=0)  # [N,14,3]
                X[edit_residue] += f_res.clamp(min=-5.0, max=5.0)[edit_residue]

            # protein-ligand external update
            protein_pos = X[mask]
            pos[mask_protein] = protein_pos
            dij = torch.norm(pos[mask_protein][external_index[0]] - pos[~mask_protein][external_index[1]], dim=1) + 1e-6
            mij = torch.cat(
                [h[mask_protein][external_index[0]], h[~mask_protein][external_index[1]], self.distance_expansion(dij)],
                dim=-1)
            xij = pos[mask_protein][external_index[0]] - pos[~mask_protein][external_index[1]]
            fij = torch.maximum(self.T_a(mij).squeeze(-1), 1.5 - dij)
            xij = xij / dij.unsqueeze(-1) * fij.unsqueeze(-1)
            f_atom = scatter_mean(xij, external_index[0], dim=0, dim_size=protein_pos.size(0))
            f_atom[batch['edit_backbone']] *= 0.1
            protein_pos += f_atom
            f_ligand_atom = scatter_mean(xij, external_index[1], dim=0, dim_size=ligand_pos.size(0))
            ligand_pos -= f_ligand_atom * 0.05

        return h, h_res, protein_pos, ligand_pos


# bilevel encoder
class BilevelEncoder(Module):
    def __init__(self, hidden_channels=128, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=8,
                 cutoff=10.0, device='cuda:0'):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.k = k
        self.cutoff = cutoff
        self.device = device
        self.esm_refine = True

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)

        # Residue level settings
        self.atom_pos_embedding = nn.Embedding(14, self.hidden_channels)
        self.residue_embedding = nn.Embedding(21, self.hidden_channels)  # one embedding for mask
        self.W_Q = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_K = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_V = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_K_lig = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_V_lig = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_O = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_O_lig = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_O_lig1 = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.act = ShiftedSoftplus()
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.layernorm1 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(0.1)

        self.T_i = nn.Sequential(nn.Linear(2 * hidden_channels + edge_channels, hidden_channels), nn.ReLU(),
                                 nn.Linear(hidden_channels, 1))
        self.T_e1 = nn.Sequential(nn.Linear(2 * hidden_channels + edge_channels, hidden_channels), nn.ReLU(),
                                 nn.Linear(hidden_channels, 1))
        self.T_e2 = nn.Sequential(nn.Linear(2 * hidden_channels + edge_channels, hidden_channels), nn.ReLU(),
                                  nn.Linear(hidden_channels, 1))
        self.residue_mlp = Linear(hidden_channels, 20)
        self.sigma_D = Sequential(
            Linear(edge_channels, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, num_heads),
        )
        self.sigma_D1 = Sequential(
            Linear(edge_channels, hidden_channels),
            nn.ReLU(),
            Linear(hidden_channels, num_heads),
        )
        self.residue_atom_mask = residue_atom_mask.to(device)

    @property
    def out_channels(self):
        return self.hidden_channels

    def connect_edges(self, res_X, batch):
        edge_index = knn_graph(res_X[:, 1], k=self.k, batch=batch, flow='target_to_source')
        edge_index, _ = add_self_loops(edge_index, num_nodes=res_X.size(0)) # add self loops
        return edge_index

    def _forward(self, res_H, res_X, res_S, batch, ligand_pos, ligand_feat, ligand_mask, edit_residue_num, residue_mask):
        atom_mask = self.residue_atom_mask[res_S]
        edge_index = self.connect_edges(res_X, batch)
        row, col = edge_index
        R_ij = torch.cdist(res_X[row], res_X[col], p=2)
        dist_rep = self.distance_expansion(R_ij).view(row.shape[0], res_X.shape[1], res_X.shape[1], -1)
        n_nodes = res_H.shape[0]
        n_edges = edge_index.shape[1]
        n_heads = self.num_heads
        n_channels = res_X.shape[1]
        d = int(self.hidden_channels / n_heads)
        res_H = self.layernorm(res_H)
        Q = self.W_Q(res_H).view([n_nodes, n_channels, n_heads, d])
        K = self.W_K(res_H).view([n_nodes, n_channels, n_heads, d])
        V = self.W_V(res_H).view([n_nodes, n_channels, n_heads, d])
        # Attention with scaled inner product
        attend_logits = torch.matmul(Q[row].transpose(1, 2), K[col].permute(0, 2, 3, 1)).view([n_edges, n_heads, n_channels, n_channels])
        attend_logits /= np.sqrt(d)
        attend_logits = attend_logits + self.sigma_D(dist_rep).permute(0, 3, 1, 2)
        attend_mask = (atom_mask[row].unsqueeze(-1) @ atom_mask[col].unsqueeze(-2))
        attend_mask = attend_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        atom_mask_head = atom_mask.unsqueeze(1).repeat(1, n_heads, 1)
        r_ij = atom_mask_head[row].unsqueeze(-2) @ attend_logits @ atom_mask_head[col].unsqueeze(-1)
        r_ij = r_ij.squeeze() / attend_mask.sum(-1).sum(-1)
        beta = scatter_softmax(r_ij, row, dim=0)

        attend_logits = torch.softmax(attend_logits, dim=-1)
        attend_logits = attend_logits * attend_mask
        attend_logits = attend_logits/(attend_logits.norm(1, dim=-1).unsqueeze(-1)+1e-7)
        alpha_ij_Vj = attend_logits @ V[col].transpose(1, 2)

        # res_H update
        update_H = scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * alpha_ij_Vj, row, dim=0, dim_size=n_nodes)
        update_H = update_H.transpose(1, 2).reshape(n_nodes, n_channels, -1) # (N, channels, heads*H_per_head)
        res_H = res_H + self.dropout(self.W_O(self.act(update_H)))
        res_H = res_H * atom_mask.unsqueeze(-1)

        # res_X update
        X_ij = res_X[row].unsqueeze(-2) - res_X[col].unsqueeze(1)
        X_ij = X_ij/(X_ij.norm(2, dim=-1).unsqueeze(-1)+1e-5)

        # Aggregate messages
        Q = Q.view(n_nodes, n_channels, -1)
        K = K.view(n_nodes, n_channels, -1)
        p_idx, q_idx = torch.cartesian_prod(torch.arange(n_channels), torch.arange(n_channels)).chunk(2, dim=-1)
        p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
        input = torch.cat([Q[row][:, p_idx].view(-1, self.hidden_channels), K[col][:, q_idx].view(-1, self.hidden_channels), dist_rep.view(-1, self.edge_channels)], dim=-1)
        f = self.dropout(self.T_i(input).view(n_edges, n_channels, n_channels))
        #attend_mask = (atom_mask[row].unsqueeze(-1) @ atom_mask[col].unsqueeze(-2)).bool()
        f = f * attend_logits.mean(1)
        res_X[residue_mask] = res_X[residue_mask] + torch.clamp(scatter_sum(beta.mean(-1).unsqueeze(-1).unsqueeze(-1) * (f.unsqueeze(-1) * X_ij).sum(-2), row, dim=0, dim_size=n_nodes)[residue_mask], min=-3.0, max=3.0)

        # consider ligand
        batch_size = batch.max().item() + 1
        lig_channel = ligand_feat.shape[1]
        row1 = torch.arange(n_nodes).to(self.device)[residue_mask]
        col1 = torch.repeat_interleave(torch.arange(batch_size).to(self.device), edit_residue_num)
        n_edges = row1.shape[0]
        Q = Q.view([n_nodes, n_channels, n_heads, d])
        ligand_feat = self.layernorm1(ligand_feat)
        K_lig = self.W_K_lig(ligand_feat).view([batch_size, lig_channel, n_heads, d])
        V_lig = self.W_V_lig(ligand_feat).view([batch_size, lig_channel, n_heads, d])
        R_ij = torch.cdist(res_X[row1], ligand_pos[col1], p=2)
        dist_rep1 = self.distance_expansion(R_ij).view(row1.shape[0], res_X.shape[1], ligand_pos.shape[1], -1)
        attend_logits = torch.matmul(Q[row1].transpose(1, 2), K_lig[col1].permute(0, 2, 3, 1)).view([n_edges, n_heads, n_channels, lig_channel])
        attend_logits /= np.sqrt(d)
        attend_logits = attend_logits + self.sigma_D1(dist_rep1).permute(0, 3, 1, 2)
        attend_mask = (atom_mask[row1].unsqueeze(-1) @ ligand_mask[col1].unsqueeze(-2))
        attend_mask = attend_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        atom_mask_head = atom_mask.unsqueeze(1).repeat(1, n_heads, 1)
        ligand_mask_head = ligand_mask.unsqueeze(1).repeat(1, n_heads, 1)
        r_ij = atom_mask_head[row1].unsqueeze(-2) @ attend_logits @ ligand_mask_head[col1].unsqueeze(-1)
        r_ij = r_ij.squeeze() / attend_mask.sum(-1).sum(-1)
        beta = scatter_softmax(r_ij, col1, dim=0)

        attend_logits_res = torch.softmax(attend_logits, dim=-1)
        attend_logits_res = attend_logits_res * attend_mask
        attend_logits_res = attend_logits_res / (attend_logits_res.norm(1, dim=-1).unsqueeze(-1) + 1e-7)
        alpha_ij_Vj = attend_logits_res @ V_lig[col1].transpose(1, 2)

        attend_logits_lig = torch.softmax(attend_logits, dim=-2) * attend_mask
        attend_logits_lig = attend_logits_lig / (attend_logits_res.norm(1, dim=-1).unsqueeze(-1) + 1e-7)

        res_H[residue_mask] = res_H[residue_mask] + self.dropout(self.W_O_lig(self.act(alpha_ij_Vj.transpose(1, 2).reshape(residue_mask.sum(), n_channels, -1))))
        res_H = res_H * atom_mask.unsqueeze(-1)
        alpha_ij_Vj_lig = attend_logits_lig.transpose(-1, -2) @ V[row1].transpose(1, 2)
        update_lig_feat = scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * alpha_ij_Vj_lig, col1, dim=0, dim_size=batch_size)
        ligand_feat = ligand_feat + self.dropout(self.W_O_lig1(self.act(update_lig_feat.transpose(1, 2).reshape(batch_size, lig_channel, -1))))

        X_ij = res_X[row1].unsqueeze(-2) - ligand_pos[col1].unsqueeze(1)
        X_ij = X_ij / (X_ij.norm(2, dim=-1).unsqueeze(-1) + 1e-5)

        Q = Q.view(n_nodes, n_channels, -1)
        K_lig = K_lig.view(batch_size, lig_channel, -1)
        p_idx, q_idx = torch.cartesian_prod(torch.arange(n_channels), torch.arange(lig_channel)).chunk(2, dim=-1)
        p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
        f_lig = self.dropout(self.T_e1(torch.cat([Q[row1][:, p_idx].view(-1, self.hidden_channels), K_lig[col1][:, q_idx].view(-1, self.hidden_channels), dist_rep1.view(-1, self.edge_channels)], dim=-1)).view(n_edges, n_channels, lig_channel))
        f_res = self.dropout(self.T_e2(torch.cat([Q[row1][:, p_idx].view(-1, self.hidden_channels), K_lig[col1][:, q_idx].view(-1, self.hidden_channels), dist_rep1.view(-1, self.edge_channels)], dim=-1)).view(n_edges, n_channels, lig_channel))
        attend_mask = (atom_mask[row1].unsqueeze(-1) @ ligand_mask[col1].unsqueeze(-2)).bool()
        f_lig = f_lig * attend_logits_lig.mean(1)
        f_res = f_res * attend_mask
        ligand_pos = ligand_pos + scatter_sum(beta.mean(-1).unsqueeze(-1).unsqueeze(-1) * (f_lig.unsqueeze(-1) * X_ij).sum(1), col1, dim=0, dim_size=batch_size)
        res_X[residue_mask] = res_X[residue_mask] + torch.clamp((f_res.unsqueeze(-1) * - X_ij).mean(-2), min=-3.0, max=3.0)

        return res_H, res_X, ligand_pos, ligand_feat

    def forward(self, res_H, res_X, res_S, batch, full_seq, ligand_pos, ligand_feat, ligand_mask, edit_residue_num, residue_mask, esmadapter, full_seq_mask, r10_mask):
        for e in range(4):
            res_H, res_X, ligand_pos, ligand_feat = self._forward(res_H, res_X, res_S, batch, ligand_pos, ligand_feat, ligand_mask, edit_residue_num, residue_mask)
        # predict residue types
        h_residue = res_H.sum(-2)
        batch_size = batch.max().item() + 1
        encoder_out = {'feats': torch.zeros(batch_size, full_seq.shape[1], self.hidden_channels).to(self.device)}
        encoder_out['feats'][r10_mask] = h_residue.view(-1, self.hidden_channels)
        init_pred = full_seq
        decode_logits = esmadapter(init_pred, encoder_out)['logits']
        pred_res_type = decode_logits[full_seq_mask][:, 4:24]
        return res_H, res_X, pred_res_type, ligand_pos


# bilevel layer
class GETLayer(Module):
    def __init__(self, hidden_channels=128, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=8,
                 cutoff=10.0, device='cuda:0', sparse_k=3):
        super().__init__()

        self.consider_ligand = False
        self.sparse_k = sparse_k

        self.hidden_channels = hidden_channels
        self.edge_channels = edge_channels
        self.key_channels = key_channels
        self.num_heads = num_heads
        self.num_interactions = num_interactions
        self.d = int(self.hidden_channels / self.num_heads)
        self.k = k
        self.cutoff = cutoff
        self.device = device
        self.esm_refine = True

        self.distance_expansion = GaussianSmearing(stop=cutoff, num_gaussians=edge_channels)

        # Residue level settings
        self.atom_pos_embedding = nn.Embedding(14, self.hidden_channels)
        self.residue_embedding = nn.Embedding(21, self.hidden_channels)  # one embedding for mask
        self.W_Q = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_K = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_V = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_Q_lig = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_K_lig = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_V_lig = nn.Linear(self.hidden_channels, self.hidden_channels, bias=False)
        self.W_O = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.W_O_lig = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.W_O_lig1 = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.act = nn.SiLU()
        self.layernorm = nn.LayerNorm(hidden_channels)
        self.layernorm1 = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(0.1)

        self.T_i = nn.Sequential(nn.Linear(3 * self.d, self.d), nn.SiLU(),
                                 nn.Linear(self.d, 1))
        self.T_e1 = nn.Sequential(nn.Linear(3 * self.d, self.d), nn.SiLU(),
                                  nn.Linear(self.d, 1))
        self.T_e2 = nn.Sequential(nn.Linear(3 * self.hidden_channels, self.hidden_channels), nn.SiLU(),
                                  nn.Linear(self.hidden_channels, self.num_heads))
        self.residue_mlp = Linear(hidden_channels, 20, bias=True)
        self.sigma_D = Sequential(
            Linear(edge_channels, hidden_channels),
            nn.SiLU(),
            Linear(hidden_channels, num_heads),
        )
        self.sigma_D1 = Sequential(
            Linear(edge_channels, hidden_channels),
            nn.SiLU(),
            Linear(hidden_channels, num_heads),
        )
        self.sigma_v = Linear(edge_channels, hidden_channels)
        self.block_mlp_invariant = nn.Sequential(nn.Linear(self.d, self.d), nn.SiLU(), nn.Linear(self.d, self.d))
        self.block_mlp_equivariant = nn.Sequential(nn.Linear(self.d, self.d), nn.SiLU(), nn.Linear(self.d, self.d))

    @property
    def out_channels(self):
        return self.hidden_channels

    def connect_edges(self, res_X, batch):
        edge_index = knn_graph(res_X[:, 1], k=self.k, batch=batch, flow='target_to_source')
        edge_index, _ = add_self_loops(edge_index, num_nodes=res_X.size(0))  # add self loops
        return edge_index

    def attention(self, res_H, res_X, atom_mask, batch, residue_mask):
        edge_index = self.connect_edges(res_X, batch)
        row, col = edge_index
        R_ij = torch.cdist(res_X[row], res_X[col], p=2)
        dist_rep = self.distance_expansion(R_ij).view(row.shape[0], res_X.shape[1], res_X.shape[1], -1)
        n_nodes = res_H.shape[0]
        n_edges = edge_index.shape[1]
        n_channels = res_X.shape[1]
        Q = self.W_Q(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        K = self.W_K(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        V = self.W_V(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        attend_logits = torch.matmul(Q[row].transpose(1, 2), K[col].permute(0, 2, 3, 1)).view(
            [n_edges, self.num_heads, n_channels, n_channels])
        attend_logits /= np.sqrt(self.d)
        attend_logits = attend_logits + self.sigma_D(dist_rep).permute(0, 3, 1, 2)  # distance bias
        attend_mask = (atom_mask[row].unsqueeze(-1) @ atom_mask[col].unsqueeze(-2)).unsqueeze(1).repeat(1,
                                                                                                        self.num_heads,
                                                                                                        1, 1)

        # sparse attention, only keep top k=3
        attend_logits[torch.logical_not(attend_mask)] = -1e5  # do not sellect from entries not attend
        _, top_indices = torch.topk(attend_logits, self.sparse_k, dim=-1, largest=True)
        sparse_mask = torch.zeros_like(attend_logits, dtype=torch.bool)
        rows = torch.arange(attend_logits.size(0)).view(-1, 1, 1, 1).expand(-1, attend_logits.size(1),
                                                                            attend_logits.size(2), self.sparse_k)
        depth = torch.arange(attend_logits.size(1)).view(1, -1, 1, 1).expand(attend_logits.size(0), -1,
                                                                             attend_logits.size(2), self.sparse_k)
        height = torch.arange(attend_logits.size(2)).view(1, 1, -1, 1).expand(attend_logits.size(0),
                                                                              attend_logits.size(1), -1, self.sparse_k)
        sparse_mask[rows, depth, height, top_indices] = True
        attend_logits = attend_logits * sparse_mask
        attend_logits = attend_logits * attend_mask

        # calculate beta
        atom_mask_head = atom_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        r_ij = atom_mask_head[row].unsqueeze(-2) @ attend_logits @ atom_mask_head[col].unsqueeze(-1)
        r_ij = r_ij.squeeze() / (attend_mask * sparse_mask).sum(-1).sum(-1)  # take avarage over non-zero entries
        beta = scatter_softmax(r_ij, row, dim=0)  # [nnodes, num_heads]

        attend_logits = torch.softmax(attend_logits, dim=-1)
        attend_logits = attend_logits * attend_mask * sparse_mask
        attend_logits = attend_logits / (
                    attend_logits.norm(1, dim=-1).unsqueeze(-1) + 1e-7)  # normalize over every rows

        alpha_ij_Vj = attend_logits @ V[col].transpose(1, 2)  # [nedges, num_heads, 14, d]
        alpha_ij_Vj = self.block_mlp_invariant(alpha_ij_Vj)

        # invariant res_H update
        update_H = scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * alpha_ij_Vj, row, dim=0, dim_size=n_nodes)
        update_H = update_H.transpose(1, 2).reshape(n_nodes, n_channels, -1)  # (nnodes, 14, heads*d)
        res_H = res_H + self.W_O(update_H)
        res_H = res_H * atom_mask.unsqueeze(-1)  # set empty entry zeros

        ###################################################################################
        # equivariant res_X update
        X_ij = res_X[row].unsqueeze(-2) - res_X[col].unsqueeze(1)  # [nedges, 14, 14, 3]
        X_ij = X_ij / (X_ij.norm(2, dim=-1).unsqueeze(-1) + 1e-5)
        X_ij = X_ij.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
        X_ij = X_ij * attend_mask.unsqueeze(-1)

        dist_rep = self.sigma_v(dist_rep).view([n_edges, n_channels, n_channels, self.num_heads, self.d]).permute(0, 3, 1, 2, 4)
        input = torch.cat(
            [(Q[row].transpose(1, 2))[rows, depth, height], (K[col].transpose(1, 2))[rows, depth, top_indices],
             dist_rep[rows, depth, height, top_indices]], dim=-1)
        f = self.T_i(input).view(n_edges, self.num_heads, n_channels, self.sparse_k)
        f = f.unsqueeze(-1) * X_ij[rows, depth, height, top_indices].view(n_edges, self.num_heads, n_channels, self.sparse_k, 3)
        f = (f * attend_logits[rows, depth, height, top_indices].unsqueeze(-1)).sum(-2)  # [nedges, num_heads, 14, 3]
        res_X[residue_mask] = res_X[residue_mask] + torch.clamp(
            scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * f, row, dim=0, dim_size=n_nodes).sum(1)[residue_mask],
            min=-3.0, max=3.0)
        res_X = res_X * atom_mask.unsqueeze(-1)  # set empty entries zeros
        return res_H, res_X

    def attention_ligand(self, res_H, res_X, atom_mask, batch, ligand_pos, ligand_feat, ligand_mask, edit_residue_num,
                         residue_mask):
        batch_size = batch.max().item() + 1
        n_nodes = res_H.shape[0]
        n_channels = res_X.shape[1]
        lig_channel = ligand_feat.shape[1]
        row = torch.arange(n_nodes).to(self.device)[residue_mask]
        col = torch.repeat_interleave(torch.arange(batch_size).to(self.device), edit_residue_num)
        n_edges = row.shape[0]
        Q = self.W_Q_lig(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        K_lig = self.W_K_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, self.d])
        V_lig = self.W_V_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, self.d])
        V = self.W_V(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        R_ij = torch.cdist(res_X[row], ligand_pos[col], p=2)
        attend_mask = (atom_mask[row].unsqueeze(-1) @ ligand_mask[col].unsqueeze(-2))
        R_ij = R_ij * attend_mask
        dist_rep1 = self.distance_expansion(R_ij).view(row.shape[0], res_X.shape[1], ligand_pos.shape[1], -1)
        attend_logits = torch.matmul(Q[row].transpose(1, 2), K_lig[col].permute(0, 2, 3, 1)).view(
            [n_edges, self.num_heads, n_channels, lig_channel])
        attend_logits /= np.sqrt(self.d)
        attend_logits = attend_logits + self.sigma_D1(dist_rep1).permute(0, 3, 1, 2)

        attend_mask = attend_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # sparse attention, only keep top k=3
        attend_logits[torch.logical_not(attend_mask)] = -1e5  # do not sellect from entries not attend
        _, top_indices = torch.topk(attend_logits, self.sparse_k, dim=-1, largest=True)
        sparse_mask = torch.zeros_like(attend_logits, dtype=torch.bool)
        rows = torch.arange(attend_logits.size(0)).view(-1, 1, 1, 1).expand(-1, attend_logits.size(1),
                                                                            attend_logits.size(2), self.sparse_k)
        depth = torch.arange(attend_logits.size(1)).view(1, -1, 1, 1).expand(attend_logits.size(0), -1,
                                                                             attend_logits.size(2), self.sparse_k)
        height = torch.arange(attend_logits.size(2)).view(1, 1, -1, 1).expand(attend_logits.size(0),
                                                                              attend_logits.size(1), -1, self.sparse_k)
        sparse_mask[rows, depth, height, top_indices] = True
        attend_logits = attend_logits * attend_mask
        attend_logits = attend_logits * sparse_mask

        # calculate beta
        atom_mask_head = atom_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        ligand_mask_head = ligand_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        r_ij = atom_mask_head[row].unsqueeze(-2) @ attend_logits @ ligand_mask_head[col].unsqueeze(-1)
        r_ij = r_ij.squeeze() / (attend_mask * sparse_mask).sum(-1).sum(-1)  # take avarage over non-zero entries
        beta = scatter_softmax(r_ij, col, dim=0)

        attend_logits_lig = torch.softmax(attend_logits, dim=-2) * attend_mask
        attend_logits_lig = attend_logits_lig / (attend_logits_lig.norm(1, dim=-2).unsqueeze(-2) + 1e-7)

        attend_logits = attend_logits * sparse_mask
        attend_logits_res = torch.softmax(attend_logits, dim=-1)
        attend_logits_res = attend_logits_res * attend_mask * sparse_mask
        attend_logits_res = attend_logits_res / (attend_logits_res.norm(1, dim=-1).unsqueeze(-1) + 1e-7)
        alpha_ij_Vj = attend_logits_res @ V_lig[col].transpose(1, 2)

        # invariant feature update
        res_H[residue_mask] = res_H[residue_mask] + self.W_O_lig(
            alpha_ij_Vj.transpose(1, 2).reshape(residue_mask.sum(), n_channels, -1))
        res_H = res_H * atom_mask.unsqueeze(-1)  # set empty entries zeros
        alpha_ij_Vj_lig = attend_logits_lig.transpose(-1, -2) @ V[row].transpose(1, 2)
        update_lig_feat = scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * alpha_ij_Vj_lig, col, dim=0,
                                      dim_size=batch_size)
        ligand_feat = ligand_feat + self.W_O_lig1(update_lig_feat.transpose(1, 2).reshape(batch_size, lig_channel, -1))
        ligand_feat = ligand_feat * ligand_mask.unsqueeze(-1)  # set empty entries zeros

        ###################################################################################
        # equivariant res_X update
        X_ij = res_X[row].unsqueeze(-2) - ligand_pos[col].unsqueeze(1)
        X_ij = X_ij / (X_ij.norm(2, dim=-1).unsqueeze(-1) + 1e-5)
        X_ij = X_ij.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
        X_ij = X_ij * attend_mask.unsqueeze(-1)

        dist_rep1 = self.sigma_v(dist_rep1).view([n_edges, n_channels, lig_channel, self.num_heads, self.d]).permute(0,
                                                                                                                     3,
                                                                                                                     1,
                                                                                                                     2,
                                                                                                                     4)
        input = torch.cat(
            [(Q[row].transpose(1, 2))[rows, depth, height], (K_lig[col].transpose(1, 2))[rows, depth, top_indices],
             dist_rep1[rows, depth, height, top_indices]], dim=-1)
        f_res = self.T_e1(input).view(n_edges, self.num_heads, n_channels, self.sparse_k)
        f_res = f_res.unsqueeze(-1) * X_ij[rows, depth, height, top_indices].view(n_edges, self.num_heads, n_channels,
                                                                                  self.sparse_k, 3)
        f_res = (f_res * attend_logits[rows, depth, height, top_indices].unsqueeze(-1)).sum(-2).sum(
            1)  # [nedges, 14, 3]
        res_X[residue_mask] = res_X[residue_mask] + torch.clamp(f_res, min=-3.0, max=3.0)
        res_X = res_X * atom_mask.unsqueeze(-1)  # set empty entries zeros

        p_idx, q_idx = torch.cartesian_prod(torch.arange(n_channels), torch.arange(lig_channel)).chunk(2, dim=-1)
        p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
        dist_rep1 = dist_rep1.permute(0, 2, 3, 1, 4)

        f_lig = self.T_e2(torch.cat(
            [Q[row][:, p_idx].view(-1, self.hidden_channels), K_lig[col][:, q_idx].view(-1, self.hidden_channels),
             dist_rep1.view(-1, self.hidden_channels)], dim=-1)).view(n_edges, n_channels, lig_channel, self.num_heads)
        # attend_mask = (atom_mask[row].unsqueeze(-1) @ ligand_mask[col].unsqueeze(-2)).bool()
        f_lig = f_lig.permute(0, 3, 1, 2) * attend_logits_lig
        f_lig = (f_lig.unsqueeze(-1) * X_ij).sum(2)  # [n_edges, num_heads, lig_channel, 3]
        ligand_pos = ligand_pos + scatter_sum((beta.unsqueeze(-1).unsqueeze(-1) * f_lig).sum(1), col, dim=0,
                                              dim_size=batch_size)
        ligand_pos = ligand_pos * ligand_mask.unsqueeze(-1)  # set empty entries zeros
        return res_H, res_X, ligand_pos, ligand_feat

    def attention_res_ligand(self, res_H, res_X, atom_mask, batch, ligand_pos, ligand_feat, ligand_mask, edit_residue_num,
                         residue_mask):
        edge_index = self.connect_edges(res_X, batch)
        row, col = edge_index
        R_ij = torch.cdist(res_X[row], res_X[col], p=2)
        dist_rep = self.distance_expansion(R_ij).view(row.shape[0], res_X.shape[1], res_X.shape[1], -1)
        n_nodes = res_H.shape[0]
        n_edges = edge_index.shape[1]
        n_channels = res_X.shape[1]
        Q = self.W_Q(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        K = self.W_K(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        V = self.W_V(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        Q_lig = self.W_Q_lig(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        attend_logits = torch.matmul(Q[row].transpose(1, 2), K[col].permute(0, 2, 3, 1)).view(
            [n_edges, self.num_heads, n_channels, n_channels])
        attend_logits /= np.sqrt(self.d)
        attend_logits = attend_logits + self.sigma_D(dist_rep).permute(0, 3, 1, 2)  # distance bias
        attend_mask = (atom_mask[row].unsqueeze(-1) @ atom_mask[col].unsqueeze(-2)).unsqueeze(1).repeat(1,
                                                                                                        self.num_heads,
                                                                                                        1, 1)

        # sparse attention, only keep top k=3
        attend_logits[torch.logical_not(attend_mask)] = -1e5  # do not sellect from entries not attend
        _, top_indices = torch.topk(attend_logits, self.sparse_k, dim=-1, largest=True)
        sparse_mask = torch.zeros_like(attend_logits, dtype=torch.bool)
        rows = torch.arange(attend_logits.size(0)).view(-1, 1, 1, 1).expand(-1, attend_logits.size(1),
                                                                            attend_logits.size(2), self.sparse_k)
        depth = torch.arange(attend_logits.size(1)).view(1, -1, 1, 1).expand(attend_logits.size(0), -1,
                                                                             attend_logits.size(2), self.sparse_k)
        height = torch.arange(attend_logits.size(2)).view(1, 1, -1, 1).expand(attend_logits.size(0),
                                                                              attend_logits.size(1), -1, self.sparse_k)
        sparse_mask[rows, depth, height, top_indices] = True
        attend_logits = attend_logits * sparse_mask
        attend_logits = attend_logits * attend_mask

        # calculate beta
        atom_mask_head = atom_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        r_ij = atom_mask_head[row].unsqueeze(-2) @ attend_logits @ atom_mask_head[col].unsqueeze(-1)
        r_ij = r_ij.squeeze() / (attend_mask * sparse_mask).sum(-1).sum(-1)  # take avarage over non-zero entries
        beta = scatter_softmax(r_ij, row, dim=0)  # [nnodes, num_heads]

        attend_logits = torch.softmax(attend_logits, dim=-1)
        attend_logits = attend_logits * attend_mask * sparse_mask
        attend_logits = attend_logits / (
                attend_logits.norm(1, dim=-1).unsqueeze(-1) + 1e-7)  # normalize over every rows

        alpha_ij_Vj = attend_logits @ V[col].transpose(1, 2)  # [nedges, num_heads, 14, d]
        alpha_ij_Vj = self.block_mlp_invariant(alpha_ij_Vj)

        # invariant res_H update
        update_H = scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * alpha_ij_Vj, row, dim=0, dim_size=n_nodes)
        update_H = update_H.transpose(1, 2).reshape(n_nodes, n_channels, -1)  # (nnodes, 14, heads*d)
        res_H = res_H + self.W_O(update_H)
        res_H = res_H * atom_mask.unsqueeze(-1)  # set empty entry zeros

        ###################################################################################
        # equivariant res_X update
        X_ij = res_X[row].unsqueeze(-2) - res_X[col].unsqueeze(1)  # [nedges, 14, 14, 3]
        X_ij = X_ij / (X_ij.norm(2, dim=-1).unsqueeze(-1) + 1e-5)
        X_ij = X_ij.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
        X_ij = X_ij * attend_mask.unsqueeze(-1)

        dist_rep = self.sigma_v(dist_rep).view([n_edges, n_channels, n_channels, self.num_heads, self.d]).permute(0, 3,
                                                                                                                  1, 2,
                                                                                                                  4)
        input = torch.cat(
            [(Q[row].transpose(1, 2))[rows, depth, height], (K[col].transpose(1, 2))[rows, depth, top_indices],
             dist_rep[rows, depth, height, top_indices]], dim=-1)
        f = self.T_i(input).view(n_edges, self.num_heads, n_channels, self.sparse_k)
        f = f.unsqueeze(-1) * X_ij[rows, depth, height, top_indices].view(n_edges, self.num_heads, n_channels,
                                                                          self.sparse_k, 3)
        f = (f * attend_logits[rows, depth, height, top_indices].unsqueeze(-1)).sum(-2)  # [nedges, num_heads, 14, 3]
        res_X[residue_mask] = res_X[residue_mask] + torch.clamp(
            scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * f, row, dim=0, dim_size=n_nodes).sum(1)[residue_mask],
            min=-3.0, max=3.0)
        res_X = res_X * atom_mask.unsqueeze(-1)  # set empty entries zeros

        ###############################################################################################################
        batch_size = batch.max().item() + 1
        n_nodes = res_H.shape[0]
        n_channels = res_X.shape[1]
        lig_channel = ligand_feat.shape[1]
        row = torch.arange(n_nodes).to(self.device)[residue_mask]
        col = torch.repeat_interleave(torch.arange(batch_size).to(self.device), edit_residue_num)
        n_edges = row.shape[0]
        #Q_lig = self.W_Q_lig(res_H).view([n_nodes, n_channels, self.num_heads, self.d])
        #Q_lig = Q
        K_lig = self.W_K_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, self.d])
        V_lig = self.W_V_lig(ligand_feat).view([batch_size, lig_channel, self.num_heads, self.d])
        R_ij = torch.cdist(res_X[row], ligand_pos[col], p=2)
        attend_mask = (atom_mask[row].unsqueeze(-1) @ ligand_mask[col].unsqueeze(-2))
        R_ij = R_ij * attend_mask
        dist_rep1 = self.distance_expansion(R_ij).view(row.shape[0], res_X.shape[1], ligand_pos.shape[1], -1)
        attend_logits = torch.matmul(Q_lig[row].transpose(1, 2), K_lig[col].permute(0, 2, 3, 1)).view(
            [n_edges, self.num_heads, n_channels, lig_channel])
        attend_logits /= np.sqrt(self.d)
        attend_logits = attend_logits + self.sigma_D1(dist_rep1).permute(0, 3, 1, 2)

        attend_mask = attend_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)

        # sparse attention, only keep top k=3
        attend_logits[torch.logical_not(attend_mask)] = -1e5  # do not sellect from entries not attend
        _, top_indices = torch.topk(attend_logits, self.sparse_k, dim=-1, largest=True)
        sparse_mask = torch.zeros_like(attend_logits, dtype=torch.bool)
        rows = torch.arange(attend_logits.size(0)).view(-1, 1, 1, 1).expand(-1, attend_logits.size(1),
                                                                            attend_logits.size(2), self.sparse_k)
        depth = torch.arange(attend_logits.size(1)).view(1, -1, 1, 1).expand(attend_logits.size(0), -1,
                                                                             attend_logits.size(2), self.sparse_k)
        height = torch.arange(attend_logits.size(2)).view(1, 1, -1, 1).expand(attend_logits.size(0),
                                                                              attend_logits.size(1), -1, self.sparse_k)
        sparse_mask[rows, depth, height, top_indices] = True
        attend_logits = attend_logits * attend_mask
        attend_logits = attend_logits * sparse_mask

        # calculate beta
        atom_mask_head = atom_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        ligand_mask_head = ligand_mask.unsqueeze(1).repeat(1, self.num_heads, 1)
        r_ij = atom_mask_head[row].unsqueeze(-2) @ attend_logits @ ligand_mask_head[col].unsqueeze(-1)
        r_ij = r_ij.squeeze() / (attend_mask * sparse_mask).sum(-1).sum(-1)  # take avarage over non-zero entries
        beta = scatter_softmax(r_ij, col, dim=0)

        attend_logits_lig = torch.softmax(attend_logits, dim=-2) * attend_mask
        attend_logits_lig = attend_logits_lig / (attend_logits_lig.norm(1, dim=-2).unsqueeze(-2) + 1e-7)

        attend_logits = attend_logits * sparse_mask
        attend_logits_res = torch.softmax(attend_logits, dim=-1)
        attend_logits_res = attend_logits_res * attend_mask * sparse_mask
        attend_logits_res = attend_logits_res / (attend_logits_res.norm(1, dim=-1).unsqueeze(-1) + 1e-7)
        alpha_ij_Vj = attend_logits_res @ V_lig[col].transpose(1, 2)

        # invariant feature update
        res_H[residue_mask] = res_H[residue_mask] + self.W_O_lig(
            alpha_ij_Vj.transpose(1, 2).reshape(residue_mask.sum(), n_channels, -1))
        res_H = res_H * atom_mask.unsqueeze(-1)  # set empty entries zeros
        alpha_ij_Vj_lig = attend_logits_lig.transpose(-1, -2) @ V[row].transpose(1, 2)
        update_lig_feat = scatter_sum(beta.unsqueeze(-1).unsqueeze(-1) * alpha_ij_Vj_lig, col, dim=0,
                                      dim_size=batch_size)
        ligand_feat = ligand_feat + self.W_O_lig1(update_lig_feat.transpose(1, 2).reshape(batch_size, lig_channel, -1))
        ligand_feat = ligand_feat * ligand_mask.unsqueeze(-1)  # set empty entries zeros

        ###################################################################################
        # equivariant res_X update
        X_ij = res_X[row].unsqueeze(-2) - ligand_pos[col].unsqueeze(1)
        X_ij = X_ij / (X_ij.norm(2, dim=-1).unsqueeze(-1) + 1e-5)
        X_ij = X_ij.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
        X_ij = X_ij * attend_mask.unsqueeze(-1)

        dist_rep1 = self.sigma_v(dist_rep1).view([n_edges, n_channels, lig_channel, self.num_heads, self.d]).permute(0,
                                                                                                                     3,
                                                                                                                     1,
                                                                                                                     2,
                                                                                                                     4)
        input = torch.cat(
            [(Q_lig[row].transpose(1, 2))[rows, depth, height], (K_lig[col].transpose(1, 2))[rows, depth, top_indices],
             dist_rep1[rows, depth, height, top_indices]], dim=-1)
        f_res = self.T_e1(input).view(n_edges, self.num_heads, n_channels, self.sparse_k)
        f_res = f_res.unsqueeze(-1) * X_ij[rows, depth, height, top_indices].view(n_edges, self.num_heads, n_channels,
                                                                                  self.sparse_k, 3)
        f_res = (f_res * attend_logits[rows, depth, height, top_indices].unsqueeze(-1)).sum(-2).sum(
            1)  # [nedges, 14, 3]
        res_X[residue_mask] = res_X[residue_mask] + torch.clamp(f_res, min=-3.0, max=3.0)
        res_X = res_X * atom_mask.unsqueeze(-1)  # set empty entries zeros

        p_idx, q_idx = torch.cartesian_prod(torch.arange(n_channels), torch.arange(lig_channel)).chunk(2, dim=-1)
        p_idx, q_idx = p_idx.squeeze(-1), q_idx.squeeze(-1)
        dist_rep1 = dist_rep1.permute(0, 2, 3, 1, 4)

        f_lig = self.T_e2(torch.cat(
            [Q_lig[row][:, p_idx].view(-1, self.hidden_channels), K_lig[col][:, q_idx].view(-1, self.hidden_channels),
             dist_rep1.view(-1, self.hidden_channels)], dim=-1)).view(n_edges, n_channels, lig_channel, self.num_heads)
        # attend_mask = (atom_mask[row].unsqueeze(-1) @ ligand_mask[col].unsqueeze(-2)).bool()
        f_lig = f_lig.permute(0, 3, 1, 2) * attend_logits_lig
        f_lig = (f_lig.unsqueeze(-1) * X_ij).sum(2)  # [n_edges, num_heads, lig_channel, 3]
        ligand_pos = ligand_pos + scatter_sum((beta.unsqueeze(-1).unsqueeze(-1) * f_lig).sum(1), col, dim=0,
                                              dim_size=batch_size)
        ligand_pos = ligand_pos * ligand_mask.unsqueeze(-1)  # set empty entries zeros
        return res_H, res_X, ligand_pos, ligand_feat

    def forward(self, res_H, res_X, atom_mask, batch, ligand_pos, ligand_feat, ligand_mask, edit_residue_num, residue_mask):
        #res_H, res_X = self.attention(res_H, res_X, atom_mask, batch, residue_mask)
        res_H, res_X, ligand_pos, ligand_feat = self.attention_res_ligand(res_H, res_X, atom_mask, batch, ligand_pos,
                                                                      ligand_feat, ligand_mask, edit_residue_num,
                                                                      residue_mask)

        pred_res_type = self.residue_mlp(res_H[residue_mask].sum(-2))
        return res_H, res_X, ligand_pos, ligand_feat, pred_res_type


class EquivariantFFN(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, n_channel=1, n_rbf=16, act_fn=nn.SiLU(),
                 residual=True, dropout=0.1, constant=1, z_requires_grad=True) -> None:
        super().__init__()
        self.constant = constant
        self.residual = residual
        self.n_rbf = n_rbf

        self.mlp_h = nn.Sequential(
            nn.Linear(d_in * 2 + n_channel * n_rbf, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_out),
            nn.Dropout(dropout)
        )

        self.mlp_z = nn.Sequential(
            nn.Linear(d_in * 2 + n_channel * n_rbf, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
            nn.Dropout(dropout)
        )

        if not z_requires_grad:
            for param in self.mlp_z.parameters():
                param.requires_grad = False

        self.rbf = RadialBasis(n_rbf, 7.0)

    def forward(self, H, X, atom_mask, residue_mask):
        '''
        :param H: [N, d_in]
        :param Z: [N, n_channel, 3]
        :param block_id: [Nu]
        '''

        radial, (X_c, X_o) = self._radial(X, atom_mask)  # [N, n_hidden_channel], ([N, 1, 3], [N, n_channel, 3]
        # H_c = scatter_mean(H, block_id, dim=0)[block_id]  # [N, d_in]
        H_c = (H * atom_mask.unsqueeze(-1)).sum(-2) / atom_mask.sum(-1).unsqueeze(-1)
        H_c = H_c.unsqueeze(-2).repeat(1, 14, 1)
        inputs = torch.cat([H, H_c, radial], dim=-1)  # [N, 14, d_in + d_in + n_rbf]

        H_update = self.mlp_h(inputs)

        H = H + H_update if self.residual else H_update

        X_update = X_c.unsqueeze(-2) + self.mlp_z(inputs) * X_o
        X[residue_mask] = X_update[residue_mask]

        H, X = H * atom_mask.unsqueeze(-1), X * atom_mask.unsqueeze(-1)

        return H, X

    def _radial(self, X, atom_mask):
        X_c = (X * atom_mask.unsqueeze(-1)).sum(-2) / atom_mask.sum(-1).unsqueeze(-1)  # center
        X_o = X - X_c.unsqueeze(-2)  # [N, 14, 3], no translation
        X_o = X_o * atom_mask.unsqueeze(-1)

        D = stable_norm(X_o, dim=-1)  # [N, 14]
        radial = self.rbf(D.view(-1)).view(D.shape[0], D.shape[1], -1)  # [N, 14, n_rbf]
        return radial, (X_c, X_o)


class InvariantLayerNorm(nn.Module):
    def __init__(self, d_hidden) -> None:
        super().__init__()

        self.layernorm = nn.LayerNorm(d_hidden)
        self.layernorm1 = nn.LayerNorm(d_hidden)

    def forward(self, res_H, ligand_feat, atom_mask, ligand_mask):
        res_H[atom_mask.bool()] = self.layernorm(res_H[atom_mask.bool()])
        ligand_feat[ligand_mask.bool()] = self.layernorm1(ligand_feat[ligand_mask.bool()])
        return res_H, ligand_feat


class GET(nn.Module):
    '''Equivariant Adaptive Block Transformer'''

    def __init__(self, hidden_channels=128, edge_channels=64, key_channels=128, num_heads=4, num_interactions=6, k=8,
                 cutoff=10.0, device='cuda:0', n_layers=4, pre_norm=False, sparse_k=3):
        super().__init__()
        '''
        :param d_hidden: Number of hidden features
        :param d_radial: Number of features for calculating geometric relations
        :param n_channel: Number of channels of coordinates of each unit
        :param n_rbf: Dimension of RBF feature, 1 for not using rbf
        :param cutoff: cutoff for RBF
        :param d_edge: Number of features for the edge features
        :param n_layers: Number of layer
        :param act_fn: Non-linearity
        :param residual: Use residual connections, we recommend not changing this one
        :param dropout: probability of dropout
        '''

        self.n_layers = n_layers
        self.pre_norm = pre_norm
        self.sparse_k = sparse_k
        self.residue_atom_mask = residue_atom_mask.to(device)

        if self.pre_norm:
            self.pre_layernorm = InvariantLayerNorm(hidden_channels)

        for i in range(0, n_layers):
            self.add_module(f'layer_{i}',
                            GETLayer(hidden_channels, edge_channels, key_channels, num_heads, num_interactions, k,
                                     cutoff, device))
            self.add_module(f'layernorm0_{i}', InvariantLayerNorm(hidden_channels))
            self.add_module(f'ffn_{i}', EquivariantFFN(
                hidden_channels, 2 * hidden_channels, hidden_channels,
            ))
            self.add_module(f'layernorm1_{i}', InvariantLayerNorm(hidden_channels))

    def forward(self, res_H, res_X, res_S, batch, ligand_pos, ligand_feat, ligand_mask, edit_residue_num, residue_mask):
        atom_mask = self.residue_atom_mask[res_S]

        if self.pre_norm:
            res_H, ligand_feat = self.pre_layernorm(res_H, ligand_feat, atom_mask, ligand_mask)

        for i in range(self.n_layers):
            res_H, res_X, ligand_pos, ligand_feat, pred_res_type = self._modules[f'layer_{i}'](res_H, res_X, atom_mask,
                                                                                               batch, ligand_pos,
                                                                                               ligand_feat, ligand_mask,
                                                                                               edit_residue_num,
                                                                                               residue_mask)
            res_H, ligand_feat = self._modules[f'layernorm0_{i}'](res_H, ligand_feat, atom_mask, ligand_mask)
            res_H, res_X = self._modules[f'ffn_{i}'](res_H, res_X, atom_mask, residue_mask)
            res_H, ligand_feat = self._modules[f'layernorm1_{i}'](res_H, ligand_feat, atom_mask, ligand_mask)

        return res_H, res_X, ligand_pos, ligand_feat, pred_res_type


def stable_norm(input, *args, **kwargs):
    return torch.norm(input, *args, **kwargs)
    input = input.clone()
    with torch.no_grad():
        sign = torch.sign(input)
        input = torch.abs(input)
        input.clamp_(min=1e-10)
        input = sign * input
    return torch.norm(input, *args, **kwargs)


if __name__ == '__main__':
    hidden_channels = 128
    edge_channels = 64
    key_channels = 128
    num_heads = 4
    device = torch.device('cuda:0')
    model = GET()
    model.to(device)
    model.eval()

    res_H = torch.rand(10, 14, hidden_channels).to(device)
    res_X = torch.rand(10, 14, 3).to(device)
    res_S = torch.ones(10, dtype=torch.long)
    atom_mask = residue_atom_mask[res_S].to(device)
    ligand_mask = torch.tensor([[1., 1, 1, 0, 0], [1, 1, 1, 1, 1]]).to(device)
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1], device=device)
    residue_mask = torch.ones(10, device=device).bool()
    edit_residue_num = torch.tensor([3, 7], device=device)
    ligand_pos, ligand_feat = torch.rand(2, 5, 3).to(device), torch.rand(2, 5, hidden_channels).to(device)
    ligand_pos = ligand_pos * ligand_mask.unsqueeze(-1)
    res_X = res_X * atom_mask.unsqueeze(-1)

    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q1, t1 = U.mm(V), torch.randn(3, device=device)
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q2, t2 = U.mm(V), torch.randn(3, device=device)

    res_H1, res_X1, ligand_pos1, ligand_feat1, pred_res_type1 = model(copy.deepcopy(res_H), res_X, res_S, batch,
                                                                      ligand_pos, copy.deepcopy(ligand_feat),
                                                                      ligand_mask, edit_residue_num,
                                                                      residue_mask)

    res_X1 = copy.deepcopy(res_X1.detach())
    res_X1[batch == 0] = torch.matmul(res_X1[batch == 0], Q1) + t1
    res_X1[batch == 1] = torch.matmul(res_X1[batch == 1], Q2) + t2
    res_X1 = res_X1 * atom_mask.unsqueeze(-1)

    ligand_pos1[0] = torch.matmul(ligand_pos1[0], Q1) + t1
    ligand_pos1[1] = torch.matmul(ligand_pos1[1], Q2) + t2
    ligand_pos1 = ligand_pos1 * ligand_mask.unsqueeze(-1)

    res_X[batch == 0] = torch.matmul(res_X[batch == 0], Q1) + t1
    res_X[batch == 1] = torch.matmul(res_X[batch == 1], Q2) + t2
    res_X = res_X * atom_mask.unsqueeze(-1)

    ligand_pos[0] = torch.matmul(ligand_pos[0], Q1) + t1
    ligand_pos[1] = torch.matmul(ligand_pos[1], Q2) + t2
    ligand_pos = ligand_pos * ligand_mask.unsqueeze(-1)

    res_H2, res_X2, ligand_pos2, ligand_feat2, pred_res_type2 = model(copy.deepcopy(res_H), res_X, res_S, batch,
                                                                      ligand_pos, copy.deepcopy(ligand_feat),
                                                                      ligand_mask, edit_residue_num,
                                                                      residue_mask)

    print((res_X1 - res_X2).norm())
    print((res_H1 - res_H2).float().norm())
    print((pred_res_type1 - pred_res_type2).norm())
    print((ligand_pos1 - ligand_pos2).norm())
    print((ligand_feat1 - ligand_feat2).norm())


