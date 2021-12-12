import torch as th
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
from datetime import datetime
from torch_geometric.transforms import Compose, ToDevice
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from copy import deepcopy
from warmup_scheduler import GradualWarmupScheduler

from data_utils import QM7Dataset, PMNetTransform
from model import PhysNetPretrain, PropertyPrediction, PMNet


parser = ArgumentParser('PhysNet')
parser.add_argument('-data_dir', type=str, default='~/.pyg/qm9')
parser.add_argument('-dataset', type=str, default='qm9')
parser.add_argument('-target_idx', type=list, default=list(range(12)))
parser.add_argument('-pretrain_batch_size', type=int, default=128)
parser.add_argument('-ckpt_step', type=int, default=1)
parser.add_argument('-ckpt_file', type=str)
parser.add_argument('-pretrain_epochs', type=int, default=50)
parser.add_argument('-lr', type=float, default=3e-4)
parser.add_argument('-cuda', action='store_true')
parser.add_argument('-pretrain', action='store_true')
parser.add_argument('-pred', action='store_true')
parser.add_argument('-pred_batch_size', type=int, default=128)
parser.add_argument('-hidden_size', type=int, default=128)
parser.add_argument('-num_pred_layers', type=int, default=3)
parser.add_argument('-pred_epochs', type=int, default=500)
parser.add_argument('-resume', action='store_true')
parser.add_argument('-resume_ckpt', type=str)


def pretrain(args):
    # create summary writer
    sw = SummaryWriter(f'logs/{args.running_id}_pretrain')
    # save arguments
    with open(f'logs/{args.running_id}_pretrain/args.json', 'w') as f:
        json.dump(vars(args), f)
    # save arguments
    data_file = os.path.join(args.data_dir, args.dataset)

    if args.dataset == 'qm7':
        dataloader = QM7Dataset(data_file)
        raise DeprecationWarning('use QM9 instead')
    elif args.dataset == 'qm9':
        qm9 = QM9(args.data_dir, transform=Compose(
            [PMNetTransform(), ToDevice(th.device('cuda') if args.cuda else th.device('cpu'))]
        ))
        dataloader = DataLoader(qm9, batch_size=args.pretrain_batch_size, shuffle=False)
    # dataloader = DataLoader(dataset, args.train_batch_size)
    model = PMNet()
    optim = Adam(model.parameters(), lr=args.lr)
    if args.cuda:
        model = model.cuda()

    mae_loss = th.nn.L1Loss()
    ce_loss = th.nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.pretrain_epochs)):
        loss_atom_types, loss_bond_types, \
        loss_bond_lengths, loss_bond_angles, loss_torsions, \
        loss_length_klds, loss_angle_klds, loss_torsion_klds, \
        total_losses = [], [], [], [], [], [], [], [], []
        for batch_idx, data in enumerate(tqdm(dataloader, leave=False)):
            model.zero_grad()
            Z, R, idx_ijk, bonds, edge_weight, \
                bond_type, bond_length, bond_angle, plane, torsion = \
            data.z, data.pos, data.idx_ijk, data.edge_index, data.edge_weight, \
                data.bond_type, data.bond_length, data.bond_angle, data.plane, data.torsion
            
            atom_type_pred, bond_type_pred, bond_length_pred, bond_angle_pred, torsion_pred, \
            loss_length_kld, loss_angle_kld, loss_torsion_kld = \
                model(Z, R, idx_ijk, bonds, edge_weight, plane)
            loss_atom_type = ce_loss(atom_type_pred, Z)
            loss_bond_type = ce_loss(bond_type_pred, bond_type)
            loss_bond_length = mae_loss(bond_length_pred, bond_length)
            loss_bond_angle = mae_loss(bond_angle_pred, bond_angle)
            loss_torsion = mae_loss(torsion_pred, torsion)
            # FIXME VAE loss learning schedule
            
            total_loss = loss_bond_type + loss_atom_type + \
                loss_bond_length + loss_bond_angle +  loss_torsion + \
                min(1, 10**(epoch//10-4)) * (loss_length_kld + loss_angle_kld + loss_torsion_kld)
            total_loss.backward()
            optim.step()

            loss_atom_types.append(loss_atom_type.detach().cpu().numpy())
            loss_bond_types.append(loss_bond_type.detach().cpu().numpy())
            loss_bond_lengths.append(loss_bond_length.detach().cpu().numpy())
            loss_bond_angles.append(loss_bond_angle.detach().cpu().numpy())
            loss_torsions.append(loss_torsion.detach().cpu().numpy())
            loss_length_klds.append(loss_length_kld.detach().cpu().numpy())
            loss_angle_klds.append(loss_angle_kld.detach().cpu().numpy())
            loss_torsion_klds.append(loss_torsion_kld.detach().cpu().numpy())
            total_losses.append(total_loss.detach().cpu().numpy())
            # for DEBUG
            if epoch == 0:
                sw.add_scalar('Debug/Atom Type', loss_atom_type, batch_idx)
                sw.add_scalar('Debug/Bond Type', loss_bond_type, batch_idx)
                sw.add_scalar('Debug/Bond Length', loss_bond_length, batch_idx)
                sw.add_scalar('Debug/Bond Angle', loss_bond_angle, batch_idx)
                sw.add_scalar('Debug/Torsion', loss_torsion, batch_idx)
                sw.add_scalar('Debug/Length KLD', loss_length_kld, batch_idx)
                sw.add_scalar('Debug/Angle KLD', loss_angle_kld, batch_idx)
                sw.add_scalar('Debug/Torsion KLD', loss_torsion_kld, batch_idx)
                sw.add_scalar('Debug/Total', total_loss, batch_idx)
 
        total = len(dataloader)
        sw.add_scalar('Loss/Atom Type', sum(loss_atom_types)/total, epoch)
        sw.add_scalar('Loss/Bond Type', sum(loss_bond_types)/total, epoch)
        sw.add_scalar('Loss/Bond Length', sum(loss_bond_lengths)/total, epoch)
        sw.add_scalar('Loss/Bond Angle', sum(loss_bond_angles)/total, epoch)
        sw.add_scalar('Loss/Torsion', sum(loss_torsions)/total, epoch)
        sw.add_scalar('Loss/Length KLD', sum(loss_length_klds)/total, epoch)
        sw.add_scalar('Loss/Angle KLD', sum(loss_angle_klds)/total, epoch)
        sw.add_scalar('Loss/Torsion KLD', sum(loss_torsion_klds)/total, epoch)
        sw.add_scalar('Loss/Total', sum(total_losses)/total, epoch)

        if (epoch+1) % args.ckpt_step == 0:
            th.save(model.state_dict(), f'logs/{args.running_id}_pretrain/epoch_%d.th' % epoch)


def run_batch(data, pretrain_model, pred_model, log, optim=None, scheduler=None, epoch=None, train=False):
    Z, R, bonds, diffusion, target = data.z, data.pos, data.edge_index, data.diffusion, data.y
    molecule_idx = th.cat([th.zeros(data.ptr[i]-data.ptr[i-1])+i-1 for i in range(1,data.ptr.shape[0])])
    molecule_idx = molecule_idx.long()
    with th.no_grad():
        h = pretrain_model.encode(Z, R, bonds, diffusion)
    if train:
        pred_model.zero_grad()
        loss = pred_model(h, molecule_idx, target)
        # clip norm
        loss.backward()
        clip_grad_norm_(pred_model.parameters(), max_norm=1000)
        optim.step()
        log.append(loss.detach().cpu())
        scheduler.step(epoch)
    else:
        with th.no_grad():
            loss = pred_model(h, molecule_idx, target)
            log.append(loss.cpu())
 

def pred(args):
    # create summary writer
    sw = SummaryWriter(f'logs/{args.running_id}_predict')
    # save arguments
    with open(f'logs/{args.running_id}_predict/args.json', 'w') as f:
        json.dump(vars(args), f)
    data_file = os.path.join(args.data_dir, args.dataset)
    if args.dataset == 'qm7':
        dataset = QM7Dataset(data_file)
        raise DeprecationWarning('use QM9 instead')
    elif args.dataset == 'qm9':
        dataset = QM9(args.data_dir, transform=Compose(
            [PMNetTransform(), ToDevice(th.device('cuda') if args.cuda else th.device('cpu'))]
        ))
    pretrain_model = PhysNetPretrain()

    pretrain_model.load_state_dict(th.load(args.ckpt_file))

    input_size = 128
    # only do single target training
    pred_model = PropertyPrediction(input_size, args.hidden_size, args.num_pred_layers)
    if args.resume:
        pred_model.load_state_dict(th.load(args.resume_ckpt))

    if args.cuda:
        pretrain_model = pretrain_model.cuda()
        pred_model = pred_model.cuda()

    optimizer = Adam(pred_model.parameters(), lr=args.lr)
    scheduler = th.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9961697)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=scheduler)

    for target_idx in args.target_idx:
        tmp_dataset = deepcopy(dataset)
        tmp_dataset.data.y = tmp_dataset.data.y[:,target_idx].unsqueeze(-1)
        if target_idx in {2,3,4,7,8,9,10}:
            # unit change, from eV to meV
            tmp_dataset.data.y = tmp_dataset.data.y * 1000
        # use same setting as before, 110k for training, next 10k for validation and last 10k for test
        train_loader = DataLoader(tmp_dataset[:110000], batch_size=args.pred_batch_size, shuffle=True)
        val_loader = DataLoader(tmp_dataset[110000:120000], batch_size=args.pred_batch_size)
        test_loader = DataLoader(tmp_dataset[120000:], batch_size=args.pred_batch_size)
        best_val_loss, best_epoch = 1e9, None
        for epoch in tqdm(range(args.pred_epochs)):
            train_losses, val_losses, test_losses = [], [], []
            for data in tqdm(train_loader, leave=False):
                run_batch(data, pretrain_model, pred_model, train_losses, optimizer,
                          scheduler_warmup, epoch, train=True)
            for data in tqdm(val_loader, leave=False):
                run_batch(data, pretrain_model, pred_model, val_losses, train=False)
            for data in tqdm(test_loader, leave=False):
                run_batch(data, pretrain_model, pred_model, test_losses, train=False)

            train_losses = th.stack(train_losses)
            test_losses = th.stack(test_losses)
            val_loss = th.stack(val_losses).mean()
        
            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
            sw.add_scalar('Validation/Target %d Best Epoch' % target_idx, best_epoch, epoch)
            sw.add_scalar('Validation/Target %d Best Valid Loss' % target_idx, best_val_loss, epoch)
            
            if target_idx == 0: sw.add_scalar('Prediction/Train μ', train_losses.mean(), epoch)
            elif target_idx == 1: sw.add_scalar('Prediction/Train α', train_losses.mean(), epoch)
            elif target_idx == 2: sw.add_scalar('Prediction/Train ε_HOMO', test_losses.mean(), epoch),
            elif target_idx == 3: sw.add_scalar('Prediction/Train ε_LUMO', test_losses.mean(), epoch),
            elif target_idx == 4: sw.add_scalar('Prediction/Train Δε', test_losses.mean(), epoch),
            elif target_idx == 5: sw.add_scalar('Prediction/Train <R>²', test_losses.mean(), epoch),
            elif target_idx == 6: sw.add_scalar('Prediction/Train ZPVE²', test_losses.mean(), epoch),
            elif target_idx == 7: sw.add_scalar('Prediction/Train U_0', test_losses.mean(), epoch),
            elif target_idx == 8: sw.add_scalar('Prediction/Train U', test_losses.mean(), epoch),
            elif target_idx == 9: sw.add_scalar('Prediction/Train H', test_losses.mean(), epoch),
            elif target_idx == 10: sw.add_scalar('Prediction/Train G', test_losses.mean(), epoch),
            elif target_idx == 11: sw.add_scalar('Prediction/Train c_v', test_losses.mean(), epoch)

            if target_idx == 0: sw.add_scalar('Prediction/Test μ', train_losses.mean(), epoch)
            elif target_idx == 1: sw.add_scalar('Prediction/Test α', train_losses.mean(), epoch)
            elif target_idx == 2: sw.add_scalar('Prediction/Test ε_HOMO', test_losses.mean(), epoch),
            elif target_idx == 3: sw.add_scalar('Prediction/Test ε_LUMO', test_losses.mean(), epoch),
            elif target_idx == 4: sw.add_scalar('Prediction/Test Δε', test_losses.mean(), epoch),
            elif target_idx == 5: sw.add_scalar('Prediction/Test <R>²', test_losses.mean(), epoch),
            elif target_idx == 6: sw.add_scalar('Prediction/Test ZPVE²', test_losses.mean(), epoch),
            elif target_idx == 7: sw.add_scalar('Prediction/Test U_0', test_losses.mean(), epoch),
            elif target_idx == 8: sw.add_scalar('Prediction/Test U', test_losses.mean(), epoch),
            elif target_idx == 9: sw.add_scalar('Prediction/Test H', test_losses.mean(), epoch),
            elif target_idx == 10: sw.add_scalar('Prediction/Test G', test_losses.mean(), epoch),
            elif target_idx == 11: sw.add_scalar('Prediction/Test c_v', test_losses.mean(), epoch)
        
            if (epoch+1) % args.ckpt_step == 0:
                th.save(pred_model.state_dict(), f'logs/{args.running_id}_predict/target_%d_epoch_%d.th' % (target_idx, epoch))


def main():
    args = parser.parse_args()
    args.running_id = '%s_%s' % (args.dataset.split('.')[0], datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M'))
    if args.pretrain:
        pretrain(args)
    if args.pred:
        pred(args)


if __name__ == '__main__':
    main()