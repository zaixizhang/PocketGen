import os
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np

from ..protein_ligand import PDBProtein, parse_sdf_file
from ..data import ProteinLigandData, torchify_dict


def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, residue_dict=None):
    instance = {}

    if protein_dict is not None:
        for key, item in protein_dict.items():
            instance['protein_' + key] = item

    if ligand_dict is not None:
        for key, item in ligand_dict.items():
            instance['ligand_' + key] = item

    if residue_dict is not None:
        for key, item in residue_dict.items():
            instance[key] = item

    return instance


class PocketLigandPairDataset(Dataset):

    def __init__(self, raw_path, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + '_processed.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + '_name2id.pt')
        self.transform = transform
        self.db = None

        self.keys = None

        if not os.path.exists(self.processed_path):
            self._process()
            self._precompute_name2id()

        self.name2id = torch.load(self.name2id_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        with open(self.index_path, 'rb') as f:
            index = pickle.load(f)

        num_skipped = 0
        with db.begin(write=True, buffers=True) as txn:
            for i, (pocket_fn, ligand_fn, _, rmsd_str) in enumerate(tqdm(index)):
                if pocket_fn is None: continue
                try:
                    pdb_data = PDBProtein(os.path.join(self.raw_path, pocket_fn))
                    pocket_dict = pdb_data.to_dict_atom()
                    residue_dict = pdb_data.to_dict_residue()
                    ligand_dict = parse_sdf_file(os.path.join(self.raw_path, ligand_fn))
                    residue_dict['protein_edit_residue'] = pdb_data.query_residues_ligand(ligand_dict)
                    assert residue_dict['protein_edit_residue'].sum()>0
                    data = from_protein_ligand_dicts(
                        protein_dict=torchify_dict(pocket_dict),
                        ligand_dict=torchify_dict(ligand_dict),
                        residue_dict=torchify_dict(residue_dict)
                    )
                    data['protein_filename'] = pocket_fn
                    data['ligand_filename'] = ligand_fn
                    txn.put(
                        key=str(i).encode(),
                        value=pickle.dumps(data)
                    )
                except:
                    num_skipped += 1
                    print('Skipping (%d) %s' % (num_skipped, ligand_fn,))
                    continue
        db.close()

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data['protein_filename'], data['ligand_filename'])
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data['id'] = idx
        assert data['protein_pos'].size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    PocketLigandPairDataset(args.path)
