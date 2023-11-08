import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import shutil
import argparse
from tqdm.auto import tqdm
import torch
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
import pickle
from kmeans_pytorch import kmeans
import numpy as np
# from torch_geometric.loader import DataLoader
from models.graphgps import bind_filter
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.mol_tree import *
from utils.transforms import *
from utils.protein_ligand import *
from torch.utils.data import DataLoader
from models.encoders import CFTransformerEncoder


def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, residue_dict=None):
    instance = {}

    if protein_dict is not None:
        for key, item in protein_dict.items():
            instance['protein_' + key] = item

    if residue_dict is not None:
        for key, item in residue_dict.items():
            instance[key] = item

    if ligand_dict is not None:
        for key, item in ligand_dict.items():
            if key == 'moltree':
                instance['moltree'] = item
            else:
                instance['ligand_' + key] = item
    return instance

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_model.yml')
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    seed_all(config.train.seed)

    # Logging
    log_dir = get_new_log_dir(args.logdir, prefix=config_name)
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'))

    # Transforms
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        FeaturizeLigandBond(),
    ])

    '''
    index = np.load('index.npy')
    vina_score = np.load('vina_score.npy')
    with open('./data/crossdocked_pocket10/index.pkl','rb') as f:
        orig_index = pickle.load(f)
    raw_path = './data/crossdocked_pocket10'
    data_list = []
    for i, ind in enumerate(tqdm(index)):
        (pocket_fn, ligand_fn, _, rmsd_str) = orig_index[ind+10000]
        vina = vina_score[i]
        if pocket_fn is None: continue
        try:
            pocket_dict = PDBProtein(os.path.join(raw_path, pocket_fn)).to_dict_atom()
            ligand_dict = parse_sdf_file(os.path.join(raw_path, ligand_fn))
            data = from_protein_ligand_dicts(
                protein_dict=torchify_dict(pocket_dict),
                ligand_dict=torchify_dict(ligand_dict)
            )
            data['protein_filename'] = pocket_fn
            data['ligand_filename'] = ligand_fn
            data['vina'] = torch.tensor([vina])
            data_list.append(data)
        except:
            continue
    f = open('data.pkl', 'wb')
    pickle.dump(data_list, f)
    f.close()
    print('Data processed!')'''

    # Datasets and loaders
    logger.info('Loading dataset...')
    with open('./data.pkl', 'rb') as f:
        dataset = pickle.load(f)
    transformed_data=[]
    for data in dataset:
        transformed_data.append(transform(data))
    train_set, val_set = transformed_data[:6800], transformed_data[-200:]
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True,
                            num_workers=4, collate_fn=collate_mols_simple)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False,
                            num_workers=4, collate_fn=collate_mols_simple)

    # Model
    logger.info('Building model...')
    #ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    model = bind_filter(
        config.model.encoder,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device).to(args.device)
    #model.load_state_dict(ckpt['model'])

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)
    loss_fn = torch.nn.MSELoss(reduction='mean')


    def train():
        model.train()
        for batch in tqdm(train_loader, desc='train'):
            for key in batch:
                batch[key] = batch[key].to(args.device)
            pred = model(protein_pos=batch['protein_pos'], protein_atom_feature=batch['protein_atom_feature'],
                         ligand_pos=batch['ligand_pos'], ligand_atom_feature=batch['ligand_atom_feature_full'],
                         batch_protein=batch['protein_element_batch'], batch_ligand=batch['ligand_element_batch'])
            optimizer.zero_grad()
            loss = loss_fn(pred, batch['vina']/10)
            loss.backward()
            optimizer.step()
        scheduler.step()


    def validate():
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                for key in batch:
                    batch[key] = batch[key].to(args.device)
                pred = model(protein_pos=batch['protein_pos'], protein_atom_feature=batch['protein_atom_feature'],
                         ligand_pos=batch['ligand_pos'], ligand_atom_feature=batch['ligand_atom_feature_full'],
                         batch_protein=batch['protein_element_batch'], batch_ligand=batch['ligand_element_batch'])
                loss = loss_fn(pred, batch['vina']/10)
                sum_loss += loss.item()
                sum_n += 1
        avg_loss = sum_loss / sum_n
        print('Validation loss:', avg_loss)
        return avg_loss


    try:
        for it in range(1, 100 + 1):
            train()
            validate()
            ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': it,
            }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
