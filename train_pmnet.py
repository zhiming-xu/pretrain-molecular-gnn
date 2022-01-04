import torch as th
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
from datetime import datetime
from torch_geometric.transforms import Compose, ToDevice, RadiusGraph
from torch_geometric.datasets import QM9, MoleculeNet, ZINC
from torch_geometric.loader import DataLoader
from warmup_scheduler import GradualWarmupScheduler

from data_utils import PMNetTransform, Scaler, DiffusionTransform
from model import PMNet, PropertyPredictionTransformer


parser = ArgumentParser('PMNet')
parser.add_argument('-data_dir', type=str, default='~/.pyg/qm9')
parser.add_argument('-dataset', type=str, default='qm9')
parser.add_argument('-target_idx', type=int, nargs='+', default=list(range(12)))
parser.add_argument('-pretrain_batch_size', type=int, default=128)
parser.add_argument('-ckpt_step', type=int, default=10)
parser.add_argument('-ckpt_file', type=str)
parser.add_argument('-pretrain_epochs', type=int, default=200)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-cuda', action='store_true')
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-pretrain', action='store_true')
parser.add_argument('-pred', type=str)
parser.add_argument('-pred_batch_size', type=int, default=128)
parser.add_argument('-hidden_size_pretrain', type=int, default=256)
parser.add_argument('-num_head', type=int, default=8)
parser.add_argument('-hidden_size_pred', type=int, default=256)
parser.add_argument('-num_pred_layers', type=int, default=3)
parser.add_argument('-pred_epochs', type=int, default=500)
parser.add_argument('-resume', action='store_true')
parser.add_argument('-resume_ckpt', type=str)

TargetName = ['μ', 'α', 'ε_HOMO', 'ε_LOMO', 'Δε', '<R>²', 'ZPVE²', 'U_0', 'U', 'H', 'G', 'c_v']


def pretrain(args):
    # create summary writer
    sw = SummaryWriter(f'logs/{args.running_id}_pretrain')
    # save arguments
    with open(f'logs/{args.running_id}_pretrain/args.json', 'w') as f:
        json.dump(vars(args), f)
    # save arguments
    data_file = os.path.join(args.data_dir, args.dataset)

    if args.dataset == 'qm9':
        qm9 = QM9(args.data_dir, transform=Compose(
            [PMNetTransform(), RadiusGraph(r=5), DiffusionTransform(), ToDevice(
                th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu')
            )]
        ))
        dataloader = DataLoader(qm9, batch_size=args.pretrain_batch_size, shuffle=False)
    # dataloader = DataLoader(dataset, args.train_batch_size)
    model = PMNet(hidden_size=args.hidden_size_pretrain, num_head=args.num_head)
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.cuda:
        device = th.device('cuda:%s' % args.gpu)
        model = model.to(device)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, verbose=True)
    scheduler_w_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0,
                                                total_epoch=10, after_scheduler=scheduler)

    mae_loss = th.nn.L1Loss()
    ce_loss = th.nn.CrossEntropyLoss()

    for epoch in tqdm(range(args.pretrain_epochs)):
        loss_atom_types, loss_bond_types, \
        loss_bond_lengths, loss_bond_angles, loss_torsions, \
        loss_length_klds, loss_angle_klds, loss_torsion_klds, \
        total_losses = [], [], [], [], [], [], [], [], []
        for batch_idx, data in enumerate(tqdm(dataloader, leave=False)):
            model.zero_grad()
            Z, R, idx_ij, idx_ijk, bonds, edge_weight, \
                bond_type, bond_length, bond_angle, plane, torsion = \
            data.z, data.pos, data.idx_ij.T, data.idx_ijk, data.edge_index, data.edge_weight, \
                data.bond_type, data.bond_length, data.bond_angle, data.plane, data.torsion
            
            atom_type_pred, bond_type_pred, bond_length_pred, bond_angle_pred, torsion_pred, \
            loss_length_kld, loss_angle_kld, loss_torsion_kld = \
                model(Z, R, idx_ij, idx_ijk, bonds, edge_weight, plane)
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
            optimizer.step()

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

        scheduler_w_warmup.step(epoch=epoch, metrics=sum(loss_torsions)/total)

        if (epoch+1) % args.ckpt_step == 0:
            th.save(model.state_dict(), f'logs/{args.running_id}_pretrain/epoch_%d.th' % epoch)


def run_batch(data, pretrain_model, pred_model, log, optim=None, scaler=None, train=False):
    Z, R, bonds, edge_weight, batch, target = data.z, data.pos, data.edge_index, data.edge_weight, data.batch, data.y
    with th.no_grad():
        h = pretrain_model.encoder(Z, bonds, pos=None, edge_weight=None)
    if train:
        pred_model.zero_grad()
        pred = pred_model(h, bonds, pos=None, edge_weight=None, batch=batch)
        # clip norm
        pred = scaler.scale_up(pred)
        loss = F.l1_loss(pred, target)
        loss.backward()
        clip_grad_norm_(pred_model.parameters(), max_norm=1000)
        optim.step()
        log.append(loss.detach().cpu())
    else:
        with th.no_grad():
            pred = pred_model(h, bonds, pos=None, edge_weight=None, batch=batch)
            pred = scaler.scale_up(pred)
            loss = F.l1_loss(pred, target)
            log.append(loss.cpu())


def run_batch_sup(data, pretrain_model, pred_model, log, optim=None, scaler=None, train=False):
    Z, R, bonds, edge_weight, batch, target = data.z, data.pos, data.edge_index, data.edge_weight, data.batch, data.y
    if train:
        pred_model.zero_grad()
        pred = pred_model(Z, bonds, batch)
        # clip norm
        pred = scaler.scale_up(pred)
        loss = F.l1_loss(pred, target)
        loss.backward()
        clip_grad_norm_(pred_model.parameters(), max_norm=1000)
        optim.step()
        log.append(loss.detach().cpu())
    else:
        with th.no_grad():
            pred = pred_model(Z, bonds, batch)
            pred = scaler.scale_up(pred)
            loss = F.l1_loss(pred, target)
            log.append(loss.cpu())


def pred_qm9(args):
    # create summary writer
    sw = SummaryWriter(f'logs/{args.running_id}_predict')
    # save arguments
    with open(f'logs/{args.running_id}_predict/args.json', 'w') as f:
        json.dump(vars(args), f)
    data_file = os.path.join(args.data_dir, args.dataset)
    targetname = ['μ', 'α', 'ε_HOMO', 'ε_LOMO', 'Δε', '<R>²', 'ZPVE²', 'U_0', 'U', 'H', 'G', 'c_v']
    pretrain_model = PMNet(hidden_size=args.hidden_size_pretrain)

    pretrain_model.load_state_dict(th.load(args.ckpt_file))

    # only do single target training
    pred_model = PropertyPredictionTransformer(args.hidden_size_pretrain, num_att_layer=args.num_pred_layers)
    if args.resume:
        pred_model.load_state_dict(th.load(args.resume_ckpt))

    if args.cuda:
        device = th.device('cuda:%s' % args.gpu)
        pretrain_model = pretrain_model.to(device)
        pred_model = pred_model.to(device)

    optimizer = Adam(pred_model.parameters(), lr=args.lr)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, verbose=True)
    scheduler_w_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=scheduler)

    if isinstance(args.target_idx, int):
        args.target_idx = [args.target_idx]
    
    for target_idx in args.target_idx:
        dataset = QM9(args.data_dir, transform=Compose(
            [PMNetTransform(), DiffusionTransform(),
            ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))]
        ))
        dataset.data.y = dataset.data.y[:,target_idx].unsqueeze(-1)
 
        scaler = Scaler(dataset.data.y)
        if scaler.scale:
            print('scale target %s' % targetname[target_idx])
        else:
            print('keep the magnitude of target %s' % targetname[target_idx])
        # use same setting as before, 110k for training, next 10k for validation and last 10k for test
        train_loader = DataLoader(dataset[:110000], batch_size=args.pred_batch_size, shuffle=True)
        val_loader = DataLoader(dataset[110000:120000], batch_size=args.pred_batch_size)
        test_loader = DataLoader(dataset[120000:], batch_size=args.pred_batch_size)
        best_val_loss, best_epoch = 1e9, None
        for epoch in tqdm(range(args.pred_epochs)):
            train_losses, val_losses, test_losses = [], [], []
            for data in tqdm(train_loader, leave=False):
                run_batch(data, pretrain_model, pred_model, train_losses, optimizer, scaler=scaler, train=True)
            for data in tqdm(val_loader, leave=False):
                run_batch(data, pretrain_model, pred_model, val_losses, scaler=scaler, train=False)
            for data in tqdm(test_loader, leave=False):
                run_batch(data, pretrain_model, pred_model, test_losses, scaler=scaler, train=False)

            train_losses = th.stack(train_losses)
            test_losses = th.stack(test_losses)
            val_loss = th.stack(val_losses).mean()
            
            scheduler_w_warmup.step(epoch, metrics=val_loss)
 
            if val_loss < best_val_loss:
                best_epoch = epoch
                best_val_loss = val_loss
            sw.add_scalar('Validation/Target %d Best Epoch' % target_idx, best_epoch, epoch)
            sw.add_scalar('Validation/Target %d Best Valid Loss' % target_idx, best_val_loss, epoch)
            
            sw.add_scalar('Prediction/Train %s' % targetname[target_idx], train_losses.mean(), epoch)
            sw.add_scalar('Prediction/Test %s' % targetname[target_idx], test_losses.mean(), epoch)
        
            if (epoch+1) % args.ckpt_step == 0:
                th.save(pred_model.state_dict(), f'logs/{args.running_id}_predict/target_%d_epoch_%d.th' % (target_idx, epoch))


# use molnet and zinc instead, do not use md17
def pred_biochem(args):
    # create summary writer
    sw = SummaryWriter(f'logs/{args.running_id}_predict')
    pretrain_model = PMNet(hidden_size=args.hidden_size_pretrain)
    pretrain_model.load_state_dict(th.load(args.ckpt_file))

    pred_model = PropertyPredictionTransformer(out_size=1)
    
    if args.resume:
        pred_model.load_state_dict(th.load(args.resume_ckpt))
    if args.cuda:
        device = th.device('cuda:%s' % args.gpu)
        pretrain_model = pretrain_model.to(device)
        pred_model = pred_model.to(device)
    
    optimizer = Adam(pred_model.parameters(), lr=args.lr)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, verbose=True)
    scheduler_w_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=scheduler)
    if args.dataset == 'HIV':
        train_dataset = MoleculeNet(args.data_dir, name=args.dataset, transform=Compose([
            DiffusionTransform(), ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))
        ]))
    elif args.dataset == 'ZINC':
        # zinc have predefined splits
        train_dataset = ZINC(args.data_dir, split='train', transform=Compose([
            DiffusionTransform(), ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))
        ]))
        val_dataset = ZINC(args.data_dir, split='val', transform=Compose([
            DiffusionTransform(), ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))
        ]))
        test_dataset = ZINC(args.data_dir, split='test', transform=Compose([
            DiffusionTransform(), ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))
        ]))
    else:
        raise NotImplementedError('no dataset %s' % args.dataset)
 
    train_loader = DataLoader(train_dataset, batch_size=args.pred_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.pred_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.pred_batch_size)
    for epoch in tqdm(range(args.pred_epochs)):
        train_losses, val_losses, test_losses = [], [], []
        for data in tqdm(train_loader, leave=False):
            run_batch_sup(data, pretrain_model, pred_model, train_losses, optimizer, train=True)
        for data in tqdm(val_loader, leave=False):
            run_batch_sup(data, pretrain_model, pred_model, val_losses, optimizer, train=False)
        for data in tqdm(test_loader, leave=False):
            run_batch_sup(data, pretrain_model, pred_model, test_losses, train=False)
        train_losses = th.stack(train_losses)
        val_losses = th.stack(val_losses)
        test_losses = th.stack(test_losses)
 
        scheduler_w_warmup.step(epoch, metrics=val_losses.mean())
        
        sw.add_scalar('Prediction/Train %s' % args.dataset, train_losses.mean(), epoch)
        sw.add_scalar('Prediction/Val %s' % args.dataset, val_losses.mean(), epoch)
        sw.add_scalar('Prediction/Test %s' % args.dataset, test_losses.mean(), epoch)
    
        if (epoch+1) % args.ckpt_step == 0:
            th.save(pred_model.state_dict(), f'logs/{args.running_id}_predict/target_%d_epoch_%d.th' % (args.dataset, epoch))


def main():
    args = parser.parse_args()
    args.running_id = '%s_%s' % (args.dataset.split('.')[0], datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M'))
    if args.pretrain:
        pretrain(args)
    
    if args.pred == 'qm':
        pred_qm9(args)
    elif args.pred == 'biochem':
        pred_biochem(args)


if __name__ == '__main__':
    main()