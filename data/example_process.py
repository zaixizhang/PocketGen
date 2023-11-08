import os
import argparse
import multiprocessing as mp
import pickle
import torch
from rdkit.Chem.rdchem import BondType
from rdkit.Chem import rdchem
import shutil

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

AA_NUMBER_NAME = {1: 'ALA', 2: 'ARG', 3: 'ASN', 4: 'ASP', 5: 'CYS', 6: 'GLN', 7: 'GLU', 8: 'GLY', 9: 'HIS', 10: 'ILE',
                  11: 'LEU', 12: 'LYS', 13: 'MET', 14: 'PHE', 15: 'PRO', 16: 'SER', 17: 'THR', 18: 'TRP', 19: 'TYR',
                  20: 'VAL'}

RES_ATOMS = [[ATOM_TYPES.index(i) for i in res if i != ''] for res in RES_ATOM14]

BOND_TYPE = {1: rdchem.BondType.SINGLE, 2: rdchem.BondType.DOUBLE, 3: rdchem.BondType.TRIPLE,
             12: rdchem.BondType.AROMATIC}

from utils.protein_ligand import PDBProtein, parse_sdf_file


def load_item(item, path):
    pdb_path = os.path.join(path, item[0])
    sdf_path = os.path.join(path, item[1])
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    with open(sdf_path, 'r') as f:
        sdf_block = f.read()
    return pdb_block, sdf_block


def process_item(item, args):
    try:
        pdb_block, sdf_block = load_item(item, args.source)
        protein = PDBProtein(pdb_block)
        seq = ''.join(protein.to_dict_residue()['seq'])
        # ligand = parse_sdf_block(sdf_block)
        ligand = parse_sdf_file(os.path.join(args.source, item[1]))

        r10_idx, r10_residues = protein.query_residues_ligand(ligand, args.radius, selected_residue=None,
                                                              return_full_seq_idx=True)
        assert len(r10_idx) == len(r10_residues)

        pdb_block_pocket = protein.residues_to_pdb_block(r10_residues)

        full_seq_idx, _ = protein.query_residues_ligand(ligand, radius=3.5, selected_residue=r10_residues,
                                                        return_full_seq_idx=True)

        ligand_fn = item[1]
        pocket_fn = ligand_fn[:-4] + '_pocket%d.pdb' % args.radius
        ligand_dest = os.path.join(args.dest, ligand_fn)
        pocket_dest = os.path.join(args.dest, pocket_fn)
        os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)

        shutil.copyfile(
            src=os.path.join(args.source, ligand_fn),
            dst=os.path.join(args.dest, ligand_fn)
        )
        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)

        return pocket_fn, ligand_fn, item[0], item[
            2], seq, full_seq_idx, r10_idx  # item[0]: original protein filename; item[2]: rmsd.

    except Exception:
        print('Exception occurred.', item)
        return None, item[1], item[0], item[2], None, None, None


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


def interpolation_init_new(residues, atoms, residue_mask):
    backbone = torch.tensor([[-0.525, 1.363, 0.0], [0.0, 0.0, 0.0], [1.526, 0.0, 0.0], [0.627, 1.062, 0.0]],)

    backbone_pos = []
    for residue in residues:
        atom_idx = residue['atoms'][1]
        backbone_pos.append([atoms[atom_idx]['x'], atoms[atom_idx]['y'], atoms[atom_idx]['z']])
    backbone_pos = torch.tensor(backbone_pos)
    res_X = torch.zeros(14, 3)

    residue_index = torch.arange(len(residue_mask))
    front = residue_index[~residue_mask][:2]
    end = residue_index[~residue_mask][-2:]
    near = nearest(residue_mask)
    for k in range(len(residue_mask)):
        if residue_mask[k]:
            res_X = []
            for idx in residues[k]['atoms']:
                res_X.append([atoms[idx]['x'], atoms[idx]['y'], atoms[idx]['z']])
            res_X = torch.tensor(res_X)
            if k < front[0]:
                alpha = backbone_pos[front[0]] + (k - front[0]) / (front[0] - front[1]) * (backbone_pos[front[0]] - backbone_pos[front[1]])
            elif k > end[1]:
                alpha = backbone_pos[end[1]] + (k - end[1]) / (end[1] - end[0]) * (backbone_pos[end[1]] - backbone_pos[end[0]])
            else:
                alpha = ((k - near[k][0]) * backbone_pos[near[k][1]] + (near[k][1] - k) * backbone_pos[near[k][0]]) * 1 / (near[k][1] - near[k][0])
            res_X[:4] = (alpha + backbone @ quaternion_to_matrix(q=torch.randn(4)).t()) * 0.3 + 0.7 * res_X[:4]
            res_X[4:] = res_X[4:] + torch.randn(res_X.shape[0]-4, 3) * 0.2
            atom_idxs = residues[k]['atoms']
            for c, idx in enumerate(atom_idxs):
                atoms[idx]['line'] = atoms[idx]['line'][:30]+str('%8.3f' % (float(res_X[c, 0]))).rjust(8) + str('%8.3f' % (float(res_X[c, 1]))).rjust(8)+ str('%8.3f' % (float(res_X[c, 2]))).rjust(8) + atoms[idx]['line'][54:]
    return residues, atoms


def to_pdb(res_X, amino_acid, res_idx, res_batch, index, original):
    lines = ['HEADER    POCKET', 'COMPND    POCKET']
    num_protein = res_batch.max().item() + 1
    for n in range(num_protein):
        mask = (res_batch == n)
        res_X_protein = res_X[mask]
        amino_acid_protein = amino_acid[mask]
        res_idx_protein = res_idx[mask]
        atom_count = 0
        if original:
            path = './data/saved/' + str(index + n) + '_orig.pdb'
        else:
            path = './data/saved/' + str(index + n) + '_gen.pdb'
        with open(path, 'w') as f:
            f.writelines(lines)
            for k in range(len(res_X_protein)):
                atom_type = RES_ATOM14[amino_acid_protein[k]]
                for i in range(NUM_ATOMS[amino_acid_protein[k]]):
                    j0 = str('ATOM').ljust(6)  # atom#6s
                    j1 = str(atom_count).rjust(5)  # aomnum#5d
                    j2 = str(atom_type[i]).center(4)  # atomname$#4s
                    j3 = AA_NUMBER_NAME[amino_acid_protein[k].item()].ljust(3)  # resname#1s
                    j4 = str('A').rjust(1)  # Astring
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
    return index + num_protein


def residues_to_pdb_block(residues, atoms, name='init'):
    block = "HEADER    %s\n" % name
    block += "COMPND    %s\n" % name
    for residue in residues:
        for atom_idx in residue['atoms']:
            block += atoms[atom_idx]['line'] + "\n"
    block += "END\n"
    return block


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='data/crossdocked_v1.1_rmsd1.0')
    args = parser.parse_args()
    pdb_path = './example_pdb.pdb'
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    protein = PDBProtein(pdb_block)
    residues, atoms = protein.return_residues()
    mask_idx = [110, 111, 185, 186, 187, 188, 189, 190, 191, 208, 210, 393, 394, 397, 398, 399, 401, 402, 405, 411, 418, 419, 420, 421, 422, 423, 424]
    mask = torch.zeros(len(residues)).bool()
    for i, res in enumerate(residues):
        if res['res_id'] in mask_idx:
            mask[i] = True
    residues, atoms = interpolation_init_new(residues, atoms, mask)
    pdb_block_pocket = residues_to_pdb_block(residues, atoms)
    with open('./inter.pdb', 'w') as f:
        f.write(pdb_block_pocket)

