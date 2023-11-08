import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from torch_geometric.nn import radius_graph, knn_graph


class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings

    def forward(self, E_idx):
        # i-j
        frequency = torch.exp(torch.arange(0, self.num_embeddings, 2, dtype=torch.float32) * -(np.log(10000.0) / self.num_embeddings)).to(E_idx.device)
        angles = E_idx.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


class ProteinFeatures(nn.Module):

    def __init__(self, num_positional_embeddings=16, num_rbf=16, top_k=8, features_type='backbone', direction='forward'):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.direction = direction

        # Feature types
        self.features_type = features_type
        self.feature_dimensions = num_positional_embeddings + num_rbf + 7

        # Positional encoding
        self.pe = PositionalEncodings(num_positional_embeddings)

    def _rbf(self, D):
        # Distance radial basis function
        D_min, D_max, D_count = 0., 20., self.num_rbf
        #D_mu = torch.linspace(D_min, D_max, D_count).cuda()
        D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)

        return RBF.squeeze(0).squeeze(0)

    def _quaternions(self, R):
        """ Convert a batch of 3D rotations [R] to quaternions [Q]
            R [...,3,3]
            Q [...,4]
        """
        # Simple Wikipedia version
        # en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
              Rxx - Ryy - Rzz, 
            - Rxx + Ryy - Rzz, 
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i,j: R[:,i,j]
        signs = torch.sign(torch.stack([
            _R(2,1) - _R(1,2),
            _R(0,2) - _R(2,0),
            _R(1,0) - _R(0,1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        # Axis of rotation
        # Replace bad rotation matrices with identity
        # I = torch.eye(3).view((1,1,1,3,3))
        # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
        # det = (
        #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
        #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
        #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
        # )
        # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
        # R = det_mask * R + (1 - det_mask) * I

        # DEBUG
        # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        # Columns of this are in rotation plane
        # A = R - I
        # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
        # axis = F.normalize(torch.cross(v1, v2), dim=-1)
        return Q

    def _contacts(self, D_neighbors, E_idx, mask_neighbors, cutoff=8):
        """ Contacts """
        D_neighbors = D_neighbors.unsqueeze(-1)
        neighbor_C = mask_neighbors * (D_neighbors < cutoff).type(torch.float32)
        return neighbor_C

    def _hbonds(self, X, E_idx, mask_neighbors, eps=1E-3):
        """ Hydrogen bonds and contact map
        """
        X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

        # Virtual hydrogens
        X_atoms['C_prev'] = F.pad(X_atoms['C'][:,1:,:], (0,0,0,1), 'constant', 0)
        X_atoms['H'] = X_atoms['N'] + F.normalize(
             F.normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
          +  F.normalize(X_atoms['N'] - X_atoms['CA'], -1)
        , -1)

        def _distance(X_a, X_b):
            return torch.norm(X_a[:,None,:,:] - X_b[:,:,None,:], dim=-1)

        def _inv_distance(X_a, X_b):
            return 1. / (_distance(X_a, X_b) + eps)

        # DSSP vacuum electrostatics model
        U = (0.084 * 332) * (
              _inv_distance(X_atoms['O'], X_atoms['N'])
            + _inv_distance(X_atoms['C'], X_atoms['H'])
            - _inv_distance(X_atoms['O'], X_atoms['H'])
            - _inv_distance(X_atoms['C'], X_atoms['N'])
        )

        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1),  E_idx)
        # print(HB)
        # HB = F.sigmoid(U)
        # U_np = U.cpu().data.numpy()
        # # plt.matshow(np.mean(U_np < -0.5, axis=0))
        # plt.matshow(HB[0,:,:])
        # plt.colorbar()
        # plt.show()
        # D_CA = _distance(X_atoms['CA'], X_atoms['CA'])
        # D_CA = D_CA.cpu().data.numpy()
        # plt.matshow(D_CA[0,:,:] < contact_D)
        # # plt.colorbar()
        # plt.show()
        # exit(0)
        return neighbor_HB

    def _AD_features(self, X, eps=1e-6):
        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1+eps, 1-eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        return F.pad(AD_features, (0,0,1,2), 'constant', 0)

    def _orientations_coarse(self, X, edge_index, residue_batch, eps=1e-6):
        # Shifted slices of unit vectors
        dX = X[1:,:] - X[:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:-1,:]
        u_1 = U[1:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        row, col = edge_index  # (E,) , (E,)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.cat([o_1, n_2, torch.cross(o_1, n_2)], dim=-1)
        set_zeros_index = torch.cumsum(residue_batch.bincount(), dim=0)[:-1]
        #O[set_zeros_index-1] = 0
        #O[set_zeros_index-2] = 0
        O = F.pad(O, (0,0,1,1), 'constant', 0)
        
        # Re-view as rotation matrices
        O = O.view(list(O.shape[:1]) + [3,3])

        # Rotate into local reference frames
        dX = X[col] - X[row]
        dU = torch.matmul(O.reshape(O.shape[0],3,3)[col], dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        R = torch.matmul(O[row], O[col].transpose(-1,-2))
        Q = self._quaternions(R)
        return torch.cat((dU,Q), dim=-1)

    def _dihedrals(self, X, eps=1e-7):
        # First 3 coordinates are N, CA, C
        X = X[:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:,1:,:] - X[:,:-1,:]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:,:-2,:]
        u_1 = U[:,1:-1,:]
        u_0 = U[:,2:,:]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1+eps, 1-eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        D = F.pad(D, (3,0), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1)/3), 3))
        phi, psi, omega = torch.unbind(D,-1)
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def forward(self, pos_ligand_coarse, edit_residue, X, S_id, batch):
        """ Featurize coordinates as an attributed graph """
        X_ca = X[:,1,:]
        edge_index = knn_graph(X_ca, k=self.top_k, batch=batch, flow='target_to_source')
        edge_length = torch.norm(X_ca[edge_index[0]] - X_ca[edge_index[1]], dim=1)
        RBF = self._rbf(edge_length)
        E_idx = S_id[edge_index[1]] - S_id[edge_index[0]]
        E_positional = self.pe(E_idx)
        O_features = self._orientations_coarse(X_ca, edge_index, batch)
        E = torch.cat([E_positional, RBF, O_features], -1)

        # additional edge index
        row = torch.arange(len(edit_residue)).to(X.device)[edit_residue]
        col = torch.cat([torch.ones(edit_residue[batch==s].sum(), dtype=torch.long)*s for s in range(batch.max().item()+1)]).to(X.device)
        edge_length_new = torch.norm(X_ca[row] - pos_ligand_coarse[col], dim=1)
        RBF = self._rbf(edge_length_new)
        E_new = torch.cat([torch.zeros(len(row), 16, device=X.device), RBF, torch.zeros(len(row), 7, device=X.device)], -1)

        return E, edge_index, edge_length, torch.cat([row.unsqueeze(0), (col+len(X)).unsqueeze(0)], 0), E_new

