import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from collections import Counter
from vina import Vina

from utils.evaluation.docking_vina import *
from utils.relax import openmm_relax, relax_sdf


if __name__ == '__main__':
    dock_score = []
    for i in range(56):
        size_factor = 1.
        buffer = 5.0
        lig_path = './data/saved/'+str(i)+'.sdf'
        pro_path = './data/saved/'+str(i)+'_gen.pdb'
        if os.path.exists(pro_path):
            try:
                print(pro_path)
                openmm_relax(pro_path)
                relax_sdf(lig_path)
                mol = Chem.MolFromMolFile(lig_path, sanitize=True)
                pos = mol.GetConformer(0).GetPositions()
                center = np.mean(pos, 0)
                ligand_pdbqt = './data/saved/'+str(i)+'.pdbqt'
                protein_pqr = './data/saved/'+str(i)+'pro.pqr'
                protein_pdbqt = './data/saved/'+str(i)+'pro.pdbqt'
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
                dock_score.append(score)
                print('Score after docking : %.3f (kcal/mol)' % score)
                v.write_poses('./data/saved/docked_'+str(i)+'.pdbqt', n_poses=1, overwrite=True)
            except:
                continue
    print('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(dock_score), np.median(dock_score)))

