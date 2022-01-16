#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
from torch_geometric.datasets import MoleculeNet, ZINC
from torch_geometric.loader import DataLoader
from warmup_scheduler import GradualWarmupScheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


from data_utils import BiochemDataset, QM9Dataset, PMNetTransform, DiffusionTransform, \
                       Scaler, scaffold_train_valid_test_split
from nn_utils import CyclicKLWeight
from model import PMNet


parser = ArgumentParser('PMNet')
parser.add_argument('-data_dir', type=str, default='~/.pyg/qm9dataset')
parser.add_argument('-dataset', type=str, default='qm9')
parser.add_argument('-target_idx', type=int, nargs='+', default=list(range(12)))
parser.add_argument('-pretrain_batch_size', type=int, default=128)
parser.add_argument('-num_pretrain_layers', type=int, default=6)
parser.add_argument('-rbf_size', type=int, default=9)
parser.add_argument('-hidden_size_pretrain', type=int, default=768)
parser.add_argument('-num_feats', type=int, default=28)
parser.add_argument('-num_elems', type=int, default=5)
parser.add_argument('-num_bond_types', type=int, default=4)
parser.add_argument('-ckpt_step', type=int, default=10)
parser.add_argument('-ckpt_file', type=str)
parser.add_argument('-pretrain_epochs', type=int, default=200)
parser.add_argument('-lr', type=float, default=1e-4)
parser.add_argument('-dropout', type=float, default=.5)
parser.add_argument('-cuda', action='store_true')
parser.add_argument('-gpu', type=int, default=0)
parser.add_argument('-pretrain', action='store_true')
parser.add_argument('-pred', type=str)
parser.add_argument('-pred_batch_size', type=int, default=128)
parser.add_argument('-num_head', type=int, default=16)
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

    if args.dataset == 'qm9':
        dataset = QM9Dataset(args.data_dir, transform=Compose(
            [PMNetTransform(), RadiusGraph(r=5), DiffusionTransform(), ToDevice(
                th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu')
            )]
        ))
        dataloader = DataLoader(dataset, batch_size=args.pretrain_batch_size, shuffle=False)
    elif args.dataset == 'HIV':
        dataset = MoleculeNet(args.data_dir, name=args.dataset, transform=Compose(
            [PMNetTransform(), RadiusGraph(r=5), DiffusionTransform(), ToDevice(
                th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu')
            )]
        ))
        dataloader = DataLoader(dataset, batch_size=args.pretrain_batch_size, shuffle=False)
    model = PMNet(hidden_size=args.hidden_size_pretrain, num_head=args.num_head, rbf_size=args.rbf_size,
                  num_att_layer=args.num_pretrain_layers, num_feats=args.num_feats,
                  num_elems=args.num_elems, num_bond_types=args.num_bond_types,
                  dropout=args.dropout, mode='pretrain')
    optimizer = Adam(model.parameters(), lr=args.lr)
    if args.cuda:
        device = th.device('cuda:%s' % args.gpu)
        model = model.to(device)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, verbose=True)
    scheduler_w_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0,
                                                total_epoch=10, after_scheduler=scheduler)

    l1_loss = th.nn.L1Loss()
    ce_loss = th.nn.CrossEntropyLoss()
    cyc = CyclicKLWeight(500)

    for epoch in tqdm(range(args.pretrain_epochs)):
        loss_atom_types, loss_bond_types, \
        loss_bond_lengths, loss_bond_angles, loss_torsions, \
        loss_length_klds, loss_angle_klds, loss_torsion_klds, \
        total_losses = [], [], [], [], [], [], [], [], []
        for batch_idx, data in enumerate(tqdm(dataloader, leave=False)):
            model.zero_grad()
            Z, X, R, idx_ij, idx_ijk, bonds, edge_weight, \
                bond_type, bond_length, bond_angle, plane, torsion = \
            data.t, data.x, data.pos, data.idx_ij.T, data.idx_ijk, data.edge_index, data.edge_weight, \
                data.bond_type, data.bond_length, data.bond_angle, data.plane, data.torsion
            
            atom_type_pred, bond_type_pred, bond_length_pred, bond_angle_pred, torsion_pred, \
            loss_length_kld, loss_angle_kld, loss_torsion_kld = \
                model(X, R, bonds, edge_weight, idx_ij=idx_ij, idx_ijk=idx_ijk, plane=plane)
            loss_atom_type = ce_loss(atom_type_pred, Z)
            loss_bond_type = ce_loss(bond_type_pred, bond_type)
            loss_bond_length = l1_loss(bond_length_pred, bond_length)
            loss_bond_angle = l1_loss(bond_angle_pred, bond_angle)
            loss_torsion = l1_loss(torsion_pred, torsion)
            
            # cyclic kl divergence schedule
            if epoch < 20:
                kld_loss = 0
            else:
                kld_loss = cyc.step() * (loss_length_kld + loss_angle_kld + loss_torsion_kld)
            total_loss = loss_bond_type + loss_atom_type + \
                loss_bond_length + loss_bond_angle +  loss_torsion + kld_loss 
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
            '''if epoch == 0:
                sw.add_scalar('Debug/Atom Type', loss_atom_type, batch_idx)
                sw.add_scalar('Debug/Bond Type', loss_bond_type, batch_idx)
                sw.add_scalar('Debug/Bond Length', loss_bond_length, batch_idx)
                sw.add_scalar('Debug/Bond Angle', loss_bond_angle, batch_idx)
                sw.add_scalar('Debug/Torsion', loss_torsion, batch_idx)
                sw.add_scalar('Debug/Length KLD', loss_length_kld, batch_idx)
                sw.add_scalar('Debug/Angle KLD', loss_angle_kld, batch_idx)
                sw.add_scalar('Debug/Torsion KLD', loss_torsion_kld, batch_idx)
                sw.add_scalar('Debug/Total', total_loss, batch_idx)'''
 
        total = len(dataloader)
        sw.add_scalar('Loss/Atom Type', sum(loss_atom_types)/total, epoch)
        sw.add_scalar('Loss/Bond Type', sum(loss_bond_types)/total, epoch)
        sw.add_scalar('Loss/Bond Length', sum(loss_bond_lengths)/total, epoch)
        sw.add_scalar('Loss/Bond Angle', sum(loss_bond_angles)/total, epoch)
        sw.add_scalar('Loss/Torsion', sum(loss_torsions)/total, epoch)
        if epoch >= 20:
            sw.add_scalar('Loss/Length KLD', sum(loss_length_klds)/total, epoch)
            sw.add_scalar('Loss/Angle KLD', sum(loss_angle_klds)/total, epoch)
            sw.add_scalar('Loss/Torsion KLD', sum(loss_torsion_klds)/total, epoch)
        sw.add_scalar('Loss/Total', sum(total_losses)/total, epoch)

        scheduler_w_warmup.step(epoch=epoch, metrics=sum(loss_torsions)/total)

        if (epoch+1) % args.ckpt_step == 0:
            th.save(model.state_dict(), f'logs/{args.running_id}_pretrain/epoch_%d.th' % epoch)


def run_batch(data, model, loss_func, log, optim=None, scaler=None, train=False):
    X, R, bonds, edge_weight, batch, target = data.x, data.pos, data.edge_index, data.edge_weight, data.batch, data.y
    if train:
        model.zero_grad()
        pred = model(X=X, R=R, bonds=bonds, edge_weight=edge_weight, batch=batch)
        if scaler is not None:
            pred = scaler.scale_up(pred)
        loss = loss_func(pred, target)
        loss.backward()
        # clip norm
        clip_grad_norm_(model.parameters(), max_norm=1000)
        optim.step()
        log[0].append(loss.detach().cpu())
    else:
        with th.no_grad():
            pred = model(X=X, R=R, bonds=bonds, edge_weight=edge_weight, batch=batch)
            if scaler is not None:
                pred = scaler.scale_up(pred)
            loss = loss_func(pred, target)
            log[0].append(loss.detach().cpu())
    if loss_func is F.binary_cross_entropy_with_logits:
        log[1] += target.detach().cpu().view(-1).tolist()
        log[2] += pred.detach().cpu().view(-1).tolist()


'''
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
'''


def pred_qm9(args):
    targetname = ['μ', 'α', 'ε_HOMO', 'ε_LOMO', 'Δε', '<R>²', 'ZPVE²', 'U_0', 'U', 'H', 'G', 'c_v']
    # create summary writer
    sw = SummaryWriter(f'logs/{args.running_id}_predict')
    # save arguments
    with open(f'logs/{args.running_id}_predict/args.json', 'w') as f:
        json.dump(vars(args), f)
    
    model = PMNet(hidden_size=args.hidden_size_pretrain, num_head=args.num_head, rbf_size=args.rbf_size,
                  num_att_layer=args.num_pretrain_layers, num_feats=args.num_feats,
                  dropout=args.dropout, mode='pred')

    pretrain_state_dict = th.load(args.ckpt_file, map_location=th.device('cpu'))
    model_state_dict = model.state_dict()
    # only load parameters in encoder and decoder
    for name, param in pretrain_state_dict.items():
        if 'generator' in name:
             continue
        if isinstance(param, th.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)

    if args.resume:
        model.load_state_dict(th.load(args.resume_ckpt, map_location=th.device('cpu')))

    if args.cuda:
        device = th.device('cuda:%s' % args.gpu)
        model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, verbose=True)
    scheduler_w_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=scheduler)

    if isinstance(args.target_idx, int):
        args.target_idx = [args.target_idx]
    
    loss_func = F.l1_loss
    for target_idx in args.target_idx:
        dataset = QM9Dataset(args.data_dir, transform=Compose(
            [PMNetTransform(), DiffusionTransform(),
            ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))]
        ))
        dataset.data.y = dataset.data.y[:,target_idx].unsqueeze(-1)
 
        scaler = Scaler(dataset.data.y)
        if scaler.scale:
            print('scale target %s' % targetname[target_idx])
        else:
            print('keep the magnitude of target %s' % targetname[target_idx])
            scaler = None
        # use same setting as before, 110k for training, next 10k for validation and last 10k for test
        train_loader = DataLoader(dataset[:110000], batch_size=args.pred_batch_size, shuffle=True)
        val_loader = DataLoader(dataset[110000:120000], batch_size=args.pred_batch_size)
        test_loader = DataLoader(dataset[120000:], batch_size=args.pred_batch_size)
        best_val_loss, best_epoch = 1e9, None
        for epoch in tqdm(range(args.pred_epochs)):
            train_losses, val_losses, test_losses = [], [], []
            for data in tqdm(train_loader, leave=False):
                run_batch(data, model, loss_func, train_losses, optimizer, scaler=scaler, train=True)
            for data in tqdm(val_loader, leave=False):
                run_batch(data, model, loss_func, val_losses, scaler=scaler, train=False)
            for data in tqdm(test_loader, leave=False):
                run_batch(data, model, loss_func, test_losses, scaler=scaler, train=False)

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
                th.save(model.state_dict(), f'logs/{args.running_id}_predict/target_%s_epoch_%d.th' % 
                        (targetname[target_idx], epoch))


# use biochem datasets, calculate pos with rdkit
def pred_biochem(args):
    # create summary writer
    sw = SummaryWriter(f'logs/{args.running_id}_predict')
    # save arguments
    with open(f'logs/{args.running_id}_predict/args.json', 'w') as f:
        json.dump(vars(args), f)
    
    model = PMNet(hidden_size=args.hidden_size_pretrain, num_head=args.num_head, rbf_size=args.rbf_size,
                  num_att_layer=args.num_pretrain_layers, num_feats=args.num_feats,
                  dropout=args.dropout, mode='pred')

    pretrain_state_dict = th.load(args.ckpt_file, map_location=th.device('cpu'))
    model_state_dict = model.state_dict()
    # only load parameters in encoder and decoder
    for name, param in pretrain_state_dict.items():
        if 'generator' in name:
             continue
        if isinstance(param, th.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)

    if args.resume:
        model.load_state_dict(th.load(args.resume_ckpt, map_location=th.device('cpu')))

    if args.cuda:
        device = th.device('cuda:%s' % args.gpu)
        model = model.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, verbose=True)
    scheduler_w_warmup = GradualWarmupScheduler(optimizer, multiplier=1.0, total_epoch=10, after_scheduler=scheduler)

    if args.dataset.lower() == 'zinc':
        # zinc have predefined splits
        train_dataset = ZINC(args.data_dir, split='train', transform=Compose([
            DiffusionTransform(), ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))
        ]))
        valid_dataset = ZINC(args.data_dir, split='val', transform=Compose([
            DiffusionTransform(), ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))
        ]))
        test_dataset = ZINC(args.data_dir, split='test', transform=Compose([
            DiffusionTransform(), ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))
        ]))
    elif args.dataset.lower() in ['freesolv', 'esol', 'lipo']:
        dataset = BiochemDataset(args.data_dir, name=args.dataset, transform=Compose([
            DiffusionTransform(), ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))
        ]))
        train_dataset, valid_dataset, test_dataset = scaffold_train_valid_test_split(dataset)
        loss_func = F.l1_loss
        is_classification = False
    elif args.dataset.lower() in ['bbbp', 'hiv']:
        dataset = BiochemDataset(args.data_dir, name=args.dataset, transform=Compose([
            DiffusionTransform(), ToDevice(th.device('cuda:%s' % args.gpu) if args.cuda else th.device('cpu'))
        ]))
        train_dataset, valid_dataset, test_dataset = scaffold_train_valid_test_split(dataset, 0.2, 0.1, 0.1)
        loss_func = F.binary_cross_entropy_with_logits
        is_classification = True
 
    train_loader = DataLoader(train_dataset, batch_size=args.pred_batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.pred_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.pred_batch_size)
 
    for epoch in tqdm(range(args.pred_epochs)):
        train_log, valid_log, test_log = [[],[],[]], [[],[],[]], [[],[],[]]
        for data in tqdm(train_loader, leave=False):
            run_batch(data, model, loss_func, train_log, optimizer, train=True)
        for data in tqdm(val_loader, leave=False):
            run_batch(data, model, loss_func, valid_log, optimizer, train=False)
        for data in tqdm(test_loader, leave=False):
            run_batch(data, model, loss_func, test_log, train=False)
        train_loss = th.stack(train_log[0])
        valid_loss = th.stack(valid_log[0])
        test_loss = th.stack(test_log[0])
        
        metrics = roc_auc_score(valid_log[1], valid_log[2]) if is_classification else valid_log.mean()
        scheduler_w_warmup.step(epoch, metrics=metrics)
        
        sw.add_scalar('Prediction-Train/%s Loss' % args.dataset, train_loss.mean(), epoch)
        sw.add_scalar('Prediction-Valid/%s Loss' % args.dataset, valid_loss.mean(), epoch)
        sw.add_scalar('Prediction-Test/%s Loss' % args.dataset, test_loss.mean(), epoch)
        if loss_func is F.binary_cross_entropy_with_logits:
            sw.add_scalar('Prediction-Train/%s AUC' % args.dataset,
                          roc_auc_score(train_log[1], train_log[2]), epoch)
            sw.add_scalar('Prediction-Valid/%s AUC' % args.dataset,
                          roc_auc_score(valid_log[1], valid_log[2]), epoch)
            sw.add_scalar('Prediction-Test/%s AUC' % args.dataset,
                          roc_auc_score(test_log[1], test_log[2]), epoch)
    
        if (epoch+1) % args.ckpt_step == 0:
            th.save(model.state_dict(), f'logs/{args.running_id}_predict/target_%s_epoch_%d.th' % (args.dataset, epoch))


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