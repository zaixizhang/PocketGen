CUDA_LAUNCH_BLOCKING=1
import shutil
import argparse
from functools import partial
import torch
torch.autograd.set_detect_anomaly(True)
import esm
from torch.nn.utils import clip_grad_norm_
import torch.utils.tensorboard
from torch_geometric.transforms import Compose
import numpy as np
from models.PD import Pocket_Design_new, init_weight
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.transforms import *
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_model.yml')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--logdir', type=str, default='./logs')
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    # Number of GPUs available
    ngpus_per_node = torch.cuda.device_count()
    device = rank % ngpus_per_node

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
    shutil.copyfile('./utils/data.py', os.path.join(log_dir, 'utils'))
    shutil.copyfile('./run.slurm', os.path.join(log_dir, 'run'))

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
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(config=config.dataset, transform=transform, )
    train_set, val_set = subsets['val'], subsets['test']
    sampler_train = DistributedSampler(train_set, num_replicas=ngpus_per_node, rank=rank)
    sampler_val = DistributedSampler(val_set, num_replicas=ngpus_per_node, rank=rank)
    train_iterator = inf_iterator(DataLoader(train_set, batch_size=config.train.batch_size,
                                             shuffle=False, num_workers=config.train.num_workers, sampler=sampler_train,
                                             collate_fn=partial(collate_mols_block, batch_converter=batch_converter)))
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False, sampler=sampler_val,
                            num_workers=config.train.num_workers, collate_fn=partial(collate_mols_block, batch_converter=batch_converter))

    # Model
    logger.info('Building model...')
    model = Pocket_Design_new(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=device
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    #ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    #model.load_state_dict(ckpt['model'])

    model.apply(init_weight)
    total = sum([param.nelement() for param in model.parameters()])
 
    print("Number of parameter: %.2fM" % (total/1e6))

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)


    def train(it):
        model.train()
        optimizer.zero_grad()
        batch = next(train_iterator)
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device)

        loss, loss_list, aar, rmsd = model(batch)
        loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        logger.info('[Train] Iter %d | Loss %.6f | Loss(huber) %.6f | Loss(pred) %.6f | Loss(bond & andgle) %.6f | AAR %.6f | RMSD %.6f '
                    '|Orig_grad_norm %.6f' % (it, loss.item(), loss_list[0].item(), loss_list[1].item(), loss_list[2], aar.item(),
                                              rmsd.item(), orig_grad_norm))
        writer.add_scalar('train/loss', loss.item(), it)
        writer.add_scalar('train/huber_loss', loss_list[0].item(), it)
        writer.add_scalar('train/pred_loss', loss_list[1].item(), it)
        writer.add_scalar('train/bondangle_loss', loss_list[2], it)
        writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
        writer.add_scalar('train/grad', orig_grad_norm, it)
        writer.flush()


    def validate(it):
        sum_loss, sum_n = 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(device)
                loss, _, _, _ = model(batch)
                sum_loss += loss.item()
                sum_n += 1
        avg_loss = sum_loss / sum_n

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info('[Validate] Iter %05d | Loss %.6f' % (it, avg_loss,))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.flush()
        return avg_loss


    try:
        for it in range(1, config.train.max_iters + 1):
            # with torch.autograd.detect_anomaly():
            train(it)
            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
        dist.destroy_process_group()
    except KeyboardInterrupt:
        logger.info('Terminating...')
