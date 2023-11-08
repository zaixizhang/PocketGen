#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import numpy as np
from scipy.spatial.transform import Rotation


# from https://github.com/charnley/rmsd/blob/master/rmsd/calculate_rmsd.py
def kabsch_rotation(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


# have been validated with kabsch from RefineGNN
def kabsch(a, b):
    # find optimal rotation matrix to transform a into b
    # a, b are both [N, 3]
    # a_aligned = aR + t
    a, b = np.array(a), np.array(b)
    a_mean = np.mean(a, axis=0)
    b_mean = np.mean(b, axis=0)
    a_c = a - a_mean
    b_c = b - b_mean

    rotation = kabsch_rotation(a_c, b_c)
    # a_aligned = np.dot(a_c, rotation)
    # t = b_mean - np.mean(a_aligned, axis=0)
    # a_aligned += t
    t = b_mean - np.dot(a_mean, rotation)
    a_aligned = np.dot(a, rotation) + t

    return a_aligned, rotation, t


# a: [N, 3], b: [N, 3]
def compute_rmsd(a, b, aligned=False):  # amino acids level rmsd
    if aligned:
        a_aligned = a
    else:
        a_aligned, _, _ = kabsch(a, b)
    dist = np.sum((a_aligned - b) ** 2, axis=-1)
    rmsd = np.sqrt(dist.sum() / a.shape[0])
    return float(rmsd)


def kabsch_torch(A, B, requires_grad=False):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    # U, S, V = torch.svd(H)
    if requires_grad:  # try more times to find a stable solution
        assert not torch.isnan(H).any()
        U, S, Vt = torch.linalg.svd(H)
        num_it = 0
        while torch.min(S) < 1e-3 or torch.min(
                torch.abs((S ** 2).view(1, 3) - (S ** 2).view(3, 1) + torch.eye(3).to(S.device))) < 1e-2:
            H = H + torch.rand(3, 3).to(H.device) * torch.eye(3).to(H.device)
            U, S, Vt = torch.linalg.svd(H)
            num_it += 1

            if num_it > 10:
                raise RuntimeError('SVD consistently numerically unstable! Exitting ... ')
    else:
        U, S, Vt = torch.linalg.svd(H)
    V = Vt.T
    # rms
    d = (torch.linalg.det(U) * torch.linalg.det(V)) < 0.0
    if d:
        SS = torch.diag(torch.tensor([1. for _ in range(len(U) - 1)] + [-1.], device=U.device, dtype=U.dtype))
        U = U @ SS
        # U[:, -1] = -U[:, -1]
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = (t.T).squeeze()
    return R.mm(A.T).T + t, R, t


def batch_kabsch_torch(A, B):
    '''
    A: [B, N, 3]
    B: [B, N, 3]
    '''
    a_mean = A.mean(dim=1, keepdims=True)
    b_mean = B.mean(dim=1, keepdims=True)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = torch.bmm(A_c.transpose(1, 2), B_c)  # [B, 3, 3]
    U, S, Vt = torch.linalg.svd(H)  # [B, 3, 3]
    V = Vt.transpose(1, 2)
    # rms
    d = ((torch.linalg.det(U) * torch.linalg.det(V)) < 0.0).long()  # [B]
    nSS = torch.diag(torch.tensor([1. for _ in range(len(U))], device=U.device, dtype=U.dtype))
    SS = torch.diag(torch.tensor([1. for _ in range(len(U) - 1)] + [-1.], device=U.device, dtype=U.dtype))
    bSS = torch.stack([nSS, SS], dim=0)[d]  # [B, 3, 3]
    U = torch.bmm(U, bSS)
    # Rotation matrix
    R = torch.bmm(V, U.transpose(1, 2))  # [B, 3, 3]
    # Translation vector
    t = b_mean - torch.bmm(R, a_mean.transpose(1, 2)).transpose(1, 2)
    A_aligned = torch.bmm(R, A.transpose(1, 2)).transpose(1, 2) + t
    return A_aligned, R, t