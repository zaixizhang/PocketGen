import sys
import numpy as np
from rdkit import Chem
import os
import logging
cwd = os.getcwd()
po_dir = os.path.abspath(os.path.join(cwd, '..'))
if not po_dir in sys.path:
    sys.path.insert(0, po_dir)

import pocketoptimizer as po
from tqdm import tqdm
import random
import shutil
from vina import Vina

import torch
sys.path.append("../../FAIR")
from utils.relax import openmm_relax, relax_sdf
from utils.protein_ligand import PDBProtein, parse_sdf_file
from utils.evaluation.docking_vina import *
# from utils.datasets.pl import PocketLigandPairDataset
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


def convert_mol2_to_sdf(mol2_filename, sdf_filename):
    # Read the mol2 file
    supplier = Chem.rdmolfiles.Mol2BlockToMol(open(mol2_filename).read())

    # Write to sdf file
    writer = Chem.rdmolfiles.SDWriter(sdf_filename)
    writer.write(supplier)
    writer.close()


def pocket_optimization(index, pro_path, lig_path, res_id, chain):
    cwd = os.getcwd()
    project_dir = os.path.join(cwd, str(index))
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(project_dir+'/designs', exist_ok=True)
    os.makedirs(project_dir + '/energies', exist_ok=True)
    os.makedirs(project_dir + '/ligand', exist_ok=True)
    os.makedirs(project_dir + '/scaffold', exist_ok=True)
    shutil.copyfile('../../../FAIR/data/saved/' + lig_path, project_dir + '/ligand/' + os.path.basename(lig_path))
    shutil.copyfile('../../../FAIR/data/saved/' + pro_path, project_dir + '/scaffold/' + os.path.basename(pro_path))

    design = po.DesignPipeline(work_dir=project_dir,
                               # Path to working directory containing scaffold and ligand subdirectory
                               ph=7,  # pH used for protein and ligand protonation
                               forcefield='amber_ff14SB',
                               # forcefield used for all energy computations (Use Amber as it is better tested!)
                               ncpus=36)
    # Prepare ligand
    design.parameterize_ligand(input_ligand='ligand/'+str(index)+'.sdf',  # Input ligand structure file could be .mol2/.sdf
                               addHs=True  # Whether to add hydrogen atoms to the input structure
                               )
    design.prepare_protein(
        protein_structure='scaffold/' + str(index)+'_whole.pdb',  # Input PDB
        keep_chains=[str(chain)],  # Specific protein chains to keep
        backbone_restraint=True,  # Restrains the backbone during the minimization
        cuda=False,  # Performs minimization on CPU instead of GPU
        discard_mols=[]
        # Special molecules to exclude. Per default everything, but peptides have to be defined manually
    )

    design.prepare_lig_conformers(
        nconfs=50,
        # Maximum number of conformers to produce (Sometimes these methods produce lower number of conformations)
        method='genetic',  # Genetic method in OpenBabel, other option is confab
        score='rmsd',  # Filters conformers based on RMSD
    )

    # Your mutations
    design.set_mutations([{'mutations': ['ALL'], 'resid': str(res_id), 'chain': str(chain)}])
    # Prepares all defined mutants and glycine scaffolds for side chain rotamer and ligand pose sampling
    design.prepare_mutants(sampling_pocket='GLY')
    # Sampling of side chain rotamers
    design.sample_sidechain_rotamers(
        vdw_filter_thresh=100,  # Energy threshold of 100 kcal/mol for filtering rotamers
        library='dunbrack',  # Use dunbrack rotamer library (Should be used!)
        dunbrack_filter_thresh=0.001,  # Probability threshold for filtering rotamers (0.1%)
        accurate=False,
        # Increases the number of rotamers sampled when using dunbrack (Be careful about the computation time!)
        include_native=True  # Include the native rotamers from the minimized structure
    )

    # Sampling of ligand poses
    # Defines a grid in which the ligand is translated and rotated along.
    #                       Range, Steps
    sample_grid = {'trans': [1, 0.5],  # Angstrom
                   'rot': [20, 20]}  # Degree
    design.sample_lig_poses(
        method='grid',  # Uses the grid method. Other option is random
        grid=sample_grid,  # Defined grid for sampling
        vdw_filter_thresh=100,  # Energy threshold of 100 kcal/mol for filtering ligand poses
        max_poses=1000  # Maximum number of poses
    )

    design.calculate_energies(
        scoring='vina',  # Method to score protein-ligand interaction
    )

    # Compute the lowest energy structures using linear programming
    design.design(
        num_solutions=3,  # Number of solutions to compute
        ligand_scaling=3,
        # Scaling factor for binding-related energies (You need to adapt this to approximate the packing and binding energies)
    )
    filepath = design.design_full_path
    shutil.copyfile(filepath+'/0/receptor.pdb', '../../../FAIR/data/saved/' + pro_path)
    lig_path = lig_path[:-4] + '.mol2'
    shutil.copyfile(filepath + '/0/ligand.mol2', '../../../FAIR/data/saved/' + lig_path)

    design.clean(scaffold=True, ligand=True)

    return pro_path, lig_path


def calculate_vina(pro_path, lig_path, i):
    size_factor = 1.
    buffer = 5.
    openmm_relax(pro_path)
    relax_sdf(lig_path)
    mol = Chem.MolFromMolFile(lig_path, sanitize=True)
    pos = mol.GetConformer(0).GetPositions()
    center = np.mean(pos, 0)
    ligand_pdbqt = './data/saved/' + str(i) + '.pdbqt'
    protein_pqr = './data/saved/' + str(i) + 'pro.pqr'
    protein_pdbqt = './data/saved/' + str(i) + 'pro.pdbqt'
    lig = PrepLig(lig_path, 'sdf')
    lig.addH()
    lig.get_pdbqt(ligand_pdbqt)

    prot = PrepProt(pro_path)
    prot.addH(protein_pqr)
    prot.get_pdbqt(protein_pdbqt)

    v = Vina(sf_name='vina', seed=0, verbosity=0)
    v.set_receptor(protein_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)
    x, y, z = (pos.max(0) - pos.min(0)) * size_factor + buffer
    v.compute_vina_maps(center=center, box_size=[x, y, z])
    energy = v.score()
    print('Score before minimization: %.3f (kcal/mol)' % energy[0])
    energy_minimized = v.optimize()
    print('Score after minimization : %.3f (kcal/mol)' % energy_minimized[0])
    v.dock(exhaustiveness=64, n_poses=30)
    score = v.energies(n_poses=1)[0][0]
    print('Score after docking : %.3f (kcal/mol)' % score)

    return score


if __name__ == '__main__':
    optimization_steps = 3
    path = './crossdocked/crossdocked_pocket10'
    #dataset = PocketLigandPairDataset(path)
    split = torch.load('../../FAIR/data/crossdocked/split.pt')
    test_idx = split['test']
    dock_score = []
    for i in tqdm(range(50)):
        lig_path = str(i)+'.sdf'
        pro_path = str(i)+'_whole.pdb'
        #try:
        print(pro_path)
        with open(pro_path, 'r') as f:
            pdb_block = f.read()
        protein = PDBProtein(pdb_block)
        residues, atoms = protein.return_residues()
        ligand = parse_sdf_file(lig_path)
        full_seq_idx, _ = protein.query_residues_ligand(ligand, radius=3.5, return_mask=False)
        # data = dataset[test_idx[i]]
        random.shuffle(full_seq_idx)
        for j in range(optimization_steps):
            chain = residues[full_seq_idx[j]]['chain']
            res_id = residues[full_seq_idx[j]]['res_id']
            assert residues[full_seq_idx[j]]['full_seq_idx'] == full_seq_idx[j]
            pro_path, lig_path = pocket_optimization(i, pro_path, lig_path, res_id, chain)

        convert_mol2_to_sdf(lig_path, lig_path[:-5] + '.sdf')

        score = calculate_vina(pro_path, lig_path, i)
        '''
        except:
            continue
        '''
