# conda install plip -c conda-forge
import shutil
import pickle
import xml.etree.ElementTree as ET
#from plip.structure.preparation import PDBComplex
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess
import os.path as osp
from tqdm import tqdm
import numpy as np
from glob import glob
import os
#from pdb_parser import PDBProtein
from Bio.PDB import PDBParser


def sdf2centroid(sdf_file):
    supp = Chem.SDMolSupplier(sdf_file, sanitize=False)
    lig_xyz = supp[0].GetConformer().GetPositions()
    centroid_x = lig_xyz[:,0].mean()
    centroid_y = lig_xyz[:,1].mean()
    centroid_z = lig_xyz[:,2].mean()
    return centroid_x, centroid_y, centroid_z

#from pdb_parser import PDBProtein
def pocket_trunction(pdb_file, threshold=10, outname=None, sdf_file=None, centroid=None):
    pdb_parser = PDBProtein(pdb_file)
    if centroid is None:
        centroid = sdf2centroid(sdf_file)
    else:
        centroid = centroid
    residues = pdb_parser.query_residues_radius(centroid,threshold)
    residue_block = pdb_parser.residues_to_pdb_block(residues)
    if outname is None:
        outname = pdb_file[:-4]+f'_pocket{threshold}.pdb'
    f = open(outname,'w')
    f.write(residue_block)
    f.close()
    return outname

def clear_plip_file(dir):
    files = glob(dir+'/plip*')
    for i in range(len(files)):
        os.remove(files[i])

def read_pkl(pkl_file):
    with open(pkl_file,'rb') as f:
        data = pickle.load(f)
    return data

def write_pkl(data_list, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(data_list, f)

def plip_parser(xml_file):
    xml_tree = ET.parse(xml_file)
    report = xml_tree.getroot()
    interaction_ele = report.findall('bindingsite/interactions')[0]
    result = {}
    for interaction in interaction_ele:
        result['num_hydrophobic'] = len(interaction_ele.findall('hydrophobic_interactions/*'))
        result['num_hydrogen'] = len(interaction_ele.findall('hydrogen_bonds/*'))
        result['num_wb'] = len(interaction_ele.findall('water_bridges/*'))
        result['num_pi_stack'] = len(interaction_ele.findall('pi_stacks/*'))
        result['num_pi_cation'] = len(interaction_ele.findall('pi_cation_interactions/*'))
        result['num_halogen'] = len(interaction_ele.findall('halogen_bonds/*'))
        result['num_metal'] = len(interaction_ele.findall('metal_complexes/*'))
    return result

def patter_analysis(ori_report, gen_report):
    compare = {}
    num_ori = 0
    num_gen = 0
    patterns = ['num_hydrophobic','num_hydrogen','num_wb','num_pi_stack','num_pi_cation','num_halogen','num_metal']
    for pattern in patterns:
        if (ori_report[pattern] == 0)&(gen_report[pattern]==0):
            continue
        num_ori += ori_report[pattern]
        num_gen += gen_report[pattern]
        #compare[pattern] = max(ori_report[pattern] - gen_report[pattern],0)
        try:
            compare[pattern] = min(gen_report[pattern]/ori_report[pattern],1)
        except:
            compare[pattern] = None

    return compare, num_ori, num_gen


def read_sdf(file):
    supp = Chem.SDMolSupplier(file)
    return [i for i in supp]

def merge_lig_pkt(pdb_file, sdf_file, out_name, mol=None):
    '''
    pdb_file = './1A1C_MALDO_2_433_0/1m4n_A_rec_1m7y_ppg_lig_tt_min_0_pocket10.pdb'
    sdf_file = './1A1C_MALDO_2_433_0/1m4n_A_rec_1m7y_ppg_lig_tt_min_0.sdf'
    '''
    protein = Chem.MolFromPDBFile(pdb_file)
    if mol == None:
        ligand = read_sdf(sdf_file)[0]
    else:
        ligand = mol
    complex = Chem.CombineMols(protein,ligand)
    Chem.MolToPDBFile(complex, out_name)

def plip_analysis(pdb_file,out_dir):
    '''
    out_dir 
    '''
    command = 'plip -f {pdb_file} -o {out_dir} -x'.format(pdb_file=pdb_file,
                                                            out_dir = out_dir)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()
    return out_dir + '/report.xml'

def plip_analysis_visual(pdb_file,out_dir):
    '''
    out_dir 
    '''
    command = 'plip -f {pdb_file} -o {out_dir} -tpy'.format(pdb_file=pdb_file,
                                                            out_dir = out_dir)
    proc = subprocess.Popen(
            command, 
            shell=True, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
    proc.communicate()
    return out_dir + '/report.xml'

def interact_analysis(results_pkl, pkt_file, sdf_file, k=10):
    '''
    Designed for a bunch of interaction analysis performed on results file
    results_pkl contained the score and docked poses
    pkt_file contained the .pdb file 
    sdf_file contained the original ligand
    '''
    results = read_pkl(results_pkl)
    scores = []
    mols = []
    for i in range(len(results)):
        try:
            scores.append(results[i][0]['affinity'])
            mols.append(results[i][0]['rdmol'])
        except:
            scores.append(0)
            mols.append(0)
    scores_zip = zip(np.sort(scores),np.argsort(scores))
    scores = np.sort(scores)
    scores_idx = np.argsort(scores)
    sorted_mols = [mols[i] for i in scores_idx]
    truncted_file = pkt_file.split('/')[-1][:-4] + '_pocket10.pdb'
    truncted_file = pocket_trunction(pkt_file, outname=f'./tmp/{truncted_file}',sdf_file=sdf_file)
    if k == 'all':
        k = len(sorted_mols)
    
    gen_report = []
    for i in range(min(k,len(sorted_mols))):
        try:
            merge_lig_pkt(truncted_file, None, f'./tmp/{i}.pdb',mol=sorted_mols[i])
            report = plip_parser(plip_analysis(f'./tmp/{i}.pdb','./tmp'))
            gen_report.append(report)
        except:
            #print(i,'failed')
            ...
    clear_plip_file('./tmp/')
    return gen_report, sdf_file.split('/')[-1]



if __name__ == '__main__':
    # make the truncted pocket, and place the correspoding molecules into them
    # note1: no conformation searching happens here, if you want to discover some
    # docked patterns, please get the docked conformations first
    # note2: the pdb file dosen't contain the ligands by default, if it has already 
    # contained ligands, please skip the merge ligand process 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb', type=str, default='./generate/luxsit/7.pdb')
    parser.add_argument('--sdf', type=str, default='./generate/luxsit/7.sdf')
    args = parser.parse_args()
        
    print(args.pdb)

    merged_pdb_file = osp.join(osp.dirname(args.pdb),'merged.pdb')
    merge_lig_pkt(args.pdb,args.sdf,merged_pdb_file)

    report = plip_parser(plip_analysis(merged_pdb_file,'interaction'))
    print(report)
    # return the pymol file for visualization
    plip_analysis_visual(merged_pdb_file,'interaction')

    
    # above is the single analysis, we want to perform some statistical analysis
    #  
