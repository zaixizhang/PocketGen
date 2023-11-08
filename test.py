CUDA_LAUNCH_BLOCKING=1
import shutil
import argparse
from tqdm.auto import tqdm
import torch
import esm
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from functools import partial
from torch_geometric.transforms import Compose
import numpy as np
from models.PD import Pocket_Design_new
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.transforms import *
from torch.utils.data import DataLoader
from collections import OrderedDict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/test_model.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Transforms
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
        #LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
    ])

    # esm
    name = 'esm1b_t33_650M_UR50S'
    pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)
    batch_converter = alphabet.get_batch_converter()

    # Datasets and loaders
    print('Loading dataset...')
    dataset, subsets = get_dataset(config=config.dataset, transform=transform, )
    train_set, test_set = subsets['train'], subsets['test']
    train_iterator = inf_iterator(DataLoader(train_set, batch_size=config.train.batch_size,
                                             shuffle=True, num_workers=config.train.num_workers,
                                             collate_fn=partial(collate_mols_block, batch_converter=batch_converter)))
    test_loader = DataLoader(test_set, batch_size=config.train.batch_size, shuffle=False,
                             num_workers=config.train.num_workers, collate_fn=partial(collate_mols_block, batch_converter=batch_converter))

    # Model
    print('Building model...')

    new_state_dict = OrderedDict()
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    '''
    for k, v in ckpt['model'].items():
        new_state_dict[k[7:]] = v'''

    model = Pocket_Design_new(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device
    ).to(args.device)
    model.load_state_dict(ckpt['model'])


    def test():
        aar_list, rmsd_list, sum_n = [], [], 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(test_loader, desc='Test'):
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(args.device)
                aar, rmsd = model.generate(batch)
                print('[Test] AAR %.6f | RMSD %.6f' % (aar.item(), rmsd.item()))
                aar_list.append(aar.item())
                rmsd_list.append(rmsd.item())
        avg_aar = np.average(aar_list)
        std_aar = np.std(aar_list)
        avg_rmsd = np.average(rmsd_list)
        std_rmsd = np.std(rmsd_list)

        print('[AVG] AAR %.6f | RMSD %.6f' % (avg_aar, avg_rmsd))
        print('[STD] AAR %.6f | RMSD %.6f' % (std_aar, std_rmsd))

    try:
        test()
    except KeyboardInterrupt:
        print('Terminating...')
