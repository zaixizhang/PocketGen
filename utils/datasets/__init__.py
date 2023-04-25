import torch
from torch.utils.data import Subset
from .pl import PocketLigandPairDataset
import random


def get_dataset(config, *args, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = PocketLigandPairDataset(root, *args, **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)
    
    if 'split' in config:
        split_by_name = torch.load(config.split)
        split = {k: [dataset.name2id[n] for n in names if n in dataset.name2id] for k, names in split_by_name.items()}
        split1 = {k: [n for n in names] for k, names in split_by_name.items()}
        torch.save(split1, 'split.pt')
        subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
        return dataset, subsets
    else:
        return dataset
