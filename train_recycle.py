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
from models.PD import Pocket_Design_new, sample_from_categorical, interpolation_init_new
from utils.datasets import *
from utils.misc import *
from utils.train import *
from utils.data import *
from utils.transforms import *
from torch.utils.data import DataLoader
import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train_model.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
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

    # Wandb
    wandb.init(
        # set the wandb project where this run will be logged
        project="pocket generation",

        # track hyperparameters and run metadata
        config=config
    )

    # Transforms
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
    ])

    # esm
    name = 'esm1b_t33_650M_UR50S'
    pretrained_model, alphabet = esm.pretrained.load_model_and_alphabet_hub(name)
    batch_converter = alphabet.get_batch_converter()
    del pretrained_model

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_dataset(config=config.dataset, transform=transform, )
    train_set, val_set = subsets['train'], subsets['test']
    train_iterator = inf_iterator(DataLoader(train_set, batch_size=config.train.batch_size,
                                             shuffle=True, num_workers=config.train.num_workers,
                                             collate_fn=partial(collate_mols_block, batch_converter=batch_converter)))
    val_loader = DataLoader(val_set, batch_size=config.train.batch_size, shuffle=False,
                            num_workers=config.train.num_workers, collate_fn=partial(collate_mols_block, batch_converter=batch_converter))

    # Model
    logger.info('Building model...')
    model = Pocket_Design_new(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        device=args.device
    ).to(args.device)
    #ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    #model.load_state_dict(ckpt['model'])

    #model.apply(init_weight)
    total = sum([param.nelement() for param in model.parameters()])

    print("Number of parameter: %.2fM" % (total/1e6))

    # Optimizer and scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    loss_list = [0., 0., 0.]
    metric_list = [0., 0.]

    def train(it, loss_list, metric_list):
        model.train()
        batch = next(train_iterator)
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(args.device)

        #loss, loss_list, aar, rmsd = model(batch)
        residue_mask = batch['protein_edit_residue']
        label_ligand = copy.deepcopy(batch['ligand_pos'])
        atom_mask = model.residue_atom_mask[batch['amino_acid'][residue_mask]].bool()
        label_X = copy.deepcopy(batch['residue_pos'])
        res_S = copy.deepcopy(batch['amino_acid_processed'])

        total_steps = torch.randint(1, 4, (1,)).item() # random sample from 1,2,3
        res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask = model.init(batch)
        for t in range(total_steps, -1, -1):
            if t == 0:
                model.train()
                res_H, res_X, ligand_pos, ligand_feat, pred_res_type = model(res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask)
            else:
                model.eval()
                with torch.no_grad():
                    res_H, res_X, ligand_pos, ligand_feat, pred_res_type = model(res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask)
        sampled_type, _ = sample_from_categorical(pred_res_type.detach())
        huber_loss = model.huber_loss(res_X[residue_mask][atom_mask], label_X[residue_mask][atom_mask]) + model.huber_loss(ligand_pos[ligand_mask.bool()], label_ligand[ligand_mask.bool()])
        pred_loss = model.pred_loss(pred_res_type, model.standard2alphabet[batch['amino_acid'][residue_mask] - 1])
        struct_loss = 2 * model.proteinloss.structure_loss(res_X[residue_mask], label_X[residue_mask], batch['amino_acid'][residue_mask] - 1, batch['res_idx'][residue_mask], batch['amino_acid_batch'][residue_mask])
        loss = huber_loss + pred_loss + struct_loss
        loss_list[0] += huber_loss
        loss_list[1] += pred_loss
        loss_list[2] += struct_loss

        aar = (model.standard2alphabet[batch['amino_acid'][residue_mask] - 1] == sampled_type).sum() / len(res_S[residue_mask])
        rmsd = torch.sqrt((res_X[residue_mask][:, :4].reshape(-1, 3) - label_X[residue_mask][:, :4].reshape(-1, 3)).norm(dim=1).sum() / len(res_S[residue_mask]) / 4)
        metric_list[0] += aar
        metric_list[1] += rmsd

        loss.backward()
        freq = 32
        if it % freq == 0:
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            total_loss = (loss_list[0] + loss_list[1] + loss_list[2]).item()/freq
            logger.info('[Train] Iter %d | Loss %.6f | Loss(huber) %.6f | Loss(pred) %.6f | Loss(bond & andgle) %.6f | AAR %.6f | RMSD %.6f '
                        '|Orig_grad_norm %.6f' % (it, total_loss, loss_list[0].item()/freq, loss_list[1].item()/freq, loss_list[2]/freq, metric_list[0].item()/freq, metric_list[1].item()/freq, orig_grad_norm))
            wandb.log({"loss": total_loss, "Loss(huber)": loss_list[0].item()/freq, "Loss(pred)": loss_list[1].item()/freq, "aar": metric_list[0].item()/freq, "rmsd": metric_list[1].item()/freq})
            writer.add_scalar('train/loss', total_loss, it)
            writer.add_scalar('train/huber_loss', loss_list[0].item()/freq, it)
            writer.add_scalar('train/pred_loss', loss_list[1].item()/freq, it)
            writer.add_scalar('train/bondangle_loss', loss_list[2]/freq, it)
            writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
            writer.add_scalar('train/grad', orig_grad_norm, it)
            writer.flush()
            loss_list = [0., 0., 0.]
            metric_list = [0., 0.]
        return loss_list, metric_list


    def validate(it):
        sum_loss, sum_n, aar, rmsd = 0, 0, 0, 0
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc='Validate'):
                for key in batch:
                    if torch.is_tensor(batch[key]):
                        batch[key] = batch[key].to(args.device)

                residue_mask = batch['protein_edit_residue']
                label_ligand = copy.deepcopy(batch['ligand_pos'])
                atom_mask = model.residue_atom_mask[batch['amino_acid'][residue_mask]].bool()
                label_X = copy.deepcopy(batch['residue_pos'])
                res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask = model.init(batch)
                for _ in range(3):
                    res_H, res_X, ligand_pos, ligand_feat, pred_res_type = model(res_H, res_X, res_S, res_batch, pred_ligand, ligand_feat, ligand_mask, edit_residue_num, residue_mask)
                ligand_mask = batch['ligand_mask'].bool()
                sampled_type, _ = sample_from_categorical(pred_res_type.detach())
                loss = model.huber_loss(res_X[residue_mask][atom_mask], label_X[residue_mask][atom_mask]) + model.huber_loss(ligand_pos[ligand_mask], label_ligand[ligand_mask])
                loss += model.pred_loss(pred_res_type, model.standard2alphabet[batch['amino_acid'][residue_mask] - 1])
                loss += 2 * model.proteinloss.structure_loss(res_X[residue_mask], label_X[residue_mask], batch['amino_acid'][residue_mask] - 1, batch['res_idx'][residue_mask], batch['amino_acid_batch'][residue_mask])
                sum_loss += loss.item()
                sum_n += 1
                aar += (model.standard2alphabet[batch['amino_acid'][residue_mask] - 1] == sampled_type).sum() / len(res_S[residue_mask])
                rmsd += torch.sqrt((res_X[residue_mask][:, :4].reshape(-1, 3) - label_X[residue_mask][:, :4].reshape(-1, 3)).norm(dim=1).sum() / len(res_S[residue_mask]) / 4)
        avg_loss = sum_loss / sum_n
        aar = aar / sum_n
        rmsd = rmsd / sum_n

        if config.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        elif config.train.scheduler.type == 'warmup_plateau':
            scheduler.step_ReduceLROnPlateau(avg_loss)
        else:
            scheduler.step()

        logger.info('[Validate] Iter %05d | Loss %.6f' % (it, avg_loss,))
        writer.add_scalar('val/loss', avg_loss, it)
        writer.add_scalar('val/aar', aar, it)
        writer.add_scalar('val/rmsd', rmsd, it)
        writer.flush()
        wandb.log(
            {"val_loss": avg_loss, "val_aar": aar, "val_rmsd": rmsd})
        return avg_loss


    try:
        for it in range(1, config.train.max_iters + 1):
            loss_list, metric_list = train(it, loss_list, metric_list)

            if it % config.train.val_freq == 0 or it == config.train.max_iters:
                validate(it)
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'iteration': it,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
        wandb.finish()
