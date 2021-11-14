import torch as th
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import os
import re
import json
from tqdm import tqdm
from datetime import datetime
from torch_geometric.transforms import Compose, ToDevice
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

from data_utils import QM7Dataset, DistanceAndPlanarAngle
from nn import PhysNetPretrain, PropertyPrediction


parser = ArgumentParser('PhysNet')
parser.add_argument('-data_dir', type=str, default='~/qm9')
parser.add_argument('-dataset', type=str, default='qm9')
parser.add_argument('-pretrain_batch_size', type=int, default=128)
parser.add_argument('-ckpt_step', type=int, default=1)
parser.add_argument('-ckpt_file', type=str)
parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-cuda', action='store_true')
parser.add_argument('-pretrain', action='store_true')
parser.add_argument('-test', action='store_true')
parser.add_argument('-pred_train_ratio', type=float, default=.8)
parser.add_argument('-pred_batch_size', type=int, default=128)
parser.add_argument('-hidden_size', type=int, default=128)
parser.add_argument('-num_pred_layers', type=int, default=3)
parser.add_argument('-pred_epochs', type=int, default=200)
parser.add_argument('-resume', action='store_true')
parser.add_argument('-resume_ckpt', type=str)


def train(args):
    # create summary writer
    sw = SummaryWriter(f'logs/{args.running_id}_pretrain')
    # save arguments
    with open(f'logs/{args.running_id}_pretrain/args.json', 'w') as f:
        json.dump(vars(args), f)
    # save arguments
    data_file = os.path.join(args.data_dir, args.dataset)

    if args.dataset == 'qm7':
        dataset = QM7Dataset(data_file)
        raise DeprecationWarning('use QM9 instead')
    elif args.dataset == 'qm9':
        qm9 = QM9(args.data_dir, transform=Compose(
            [DistanceAndPlanarAngle(), ToDevice(th.device('cuda') if args.cuda else th.device('cpu'))]
        ))
        dataset = DataLoader(qm9, batch_size=args.pretrain_batch_size, shuffle=True)
    # dataloader = DataLoader(dataset, args.train_batch_size)
    model = PhysNetPretrain()
    optim = SGD(model.parameters(), lr=args.lr)
    if args.cuda:
        model = model.cuda()
        # dataset = dataset.cuda()

    for epoch in tqdm(range(args.num_epochs)):
        loss_atom_types, loss_bond_types, \
        loss_bond_lengths, loss_bond_angles, loss_torsions, \
        loss_length_klds, loss_angle_klds, loss_torsion_klds, \
        total_losses = [], [], [], [], [], [], [], [], []
        for batch_idx, data in enumerate(tqdm(dataset, leave=False)):
            model.zero_grad()
            Z, R, idx_ijk, bonds, \
                bond_type, bond_length, bond_angle, plane, torsion = \
            data.z, data.pos, data.idx_ijk, data.edge_index, \
                data.bond_type, data.bond_length, data.bond_angle, data.plane, data.torsion
            
            loss_atom_type, loss_bond_type, \
            loss_bond_length, loss_bond_angle, loss_torsion, \
            loss_length_kld, loss_angle_kld, loss_torsion_kld = \
                model(Z, R, idx_ijk, bonds, bond_type, bond_length, bond_angle, plane, torsion)
            # FIXME VAE loss learning schedule
            
            total_loss = loss_bond_type + loss_atom_type + \
                loss_bond_length + loss_bond_angle +  loss_torsion, \
                ((epoch//3+1)*0.07) * (loss_length_kld + loss_angle_kld + loss_torsion_kld)
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
            sw.add_scalar('Debug/Atom Type', loss_atom_type, batch_idx)
            sw.add_scalar('Debug/Bond Type', loss_bond_type, batch_idx)
            sw.add_scalar('Debug/Bond Length', loss_bond_length, batch_idx)
            sw.add_scalar('Debug/Bond Angle', loss_bond_angle, batch_idx)
            sw.add_scalar('Debug/Torsion', loss_torsion, batch_idx)
            sw.add_scalar('Debug/Length KLD', loss_length_kld, batch_idx)
            sw.add_scalar('Debug/Angle KLD', loss_angle_kld, batch_idx)
            sw.add_scalar('Debug/Torsion KLD', loss_torsion_kld, batch_idx)
            sw.add_scalar('Debug/Total', total_loss, batch_idx)
 
        total = len(dataset)
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
        qm9 = QM9(args.data_dir, transform=Compose(
            [DistanceAndPlanarAngle(), ToDevice(th.device('cuda') if args.cuda else th.device('cpu'))]
        ))
        # qm9.data.y = (qm9.data.y-qm9.data.y.mean(dim=0)) / qm9.data.y.std(dim=0)
        qm9.data.y = qm9.data.y[:,:12] # only use the first 12 targets
        dataset = DataLoader(qm9, batch_size=args.pred_batch_size)
    # dataloader = DataLoader(dataset, args.train_batch_size)
    pretrain_model = PhysNetPretrain()

    pretrain_model.load_state_dict(th.load(args.ckpt_file))

    input_size = 128
    pred_model = PropertyPrediction(input_size, args.hidden_size, args.num_pred_layers, target_size=12)
    if args.resume:
        pred_model.load_state_dict(th.load(args.resume_ckpt))

    if args.cuda:
        pretrain_model = pretrain_model.cuda()
        pred_model = pred_model.cuda()

    # optimizer1 = SGD(pretrain_model.parameters(), lr=1e-6)
    optimizer2 = Adam(pred_model.parameters(), lr=3e-4)

    total = len(dataset)
    num_train = round(total * args.pred_train_ratio)
    for epoch in tqdm(range(args.pred_epochs)):
        train_losses, test_losses = [], []
        cnt = 0
        for data in tqdm(dataset, leave=False):
            Z, R, bonds, target = data.z, data.pos, data.edge_index, data.y
            molecule_idx = th.cat([th.zeros(data.ptr[i]-data.ptr[i-1])+i-1 for i in range(1,data.ptr.shape[0])])
            molecule_idx = molecule_idx.long()
            with th.no_grad():
                h = pretrain_model.encode(Z, R, bonds)
            if cnt < num_train:
                pred_model.zero_grad()
                loss = pred_model(h, molecule_idx, data.y)
                loss.mean().backward()
                # optimizer1.step()
                optimizer2.step()
                train_losses.append(loss.detach().cpu().mean(dim=0))
            else:
                with th.no_grad():
                    h = pretrain_model.encode(Z, R, bonds)
                    loss = pred_model(h, molecule_idx, target)
                    test_losses.append(loss.detach().cpu().mean(dim=0))
            cnt += 1

        train_losses = th.stack(train_losses)
        test_losses = th.stack(test_losses)
        
        sw.add_scalar('Prediction/Train μ', train_losses[:,0].mean(), epoch)
        sw.add_scalar('Prediction/Train α', train_losses[:,1].mean(), epoch)
        sw.add_scalar('Prediction/Train ε_HOMO', test_losses[:,2].mean(), epoch)
        sw.add_scalar('Prediction/Train ε_LUMO', test_losses[:,3].mean(), epoch)
        sw.add_scalar('Prediction/Train Δε', test_losses[:,4].mean(), epoch)
        sw.add_scalar('Prediction/Train <R>²', test_losses[:,5].mean(), epoch)
        sw.add_scalar('Prediction/Train ZPVE²', test_losses[:,6].mean(), epoch)
        sw.add_scalar('Prediction/Train U_0', test_losses[:,7].mean(), epoch)
        sw.add_scalar('Prediction/Train U', test_losses[:,8].mean(), epoch)
        sw.add_scalar('Prediction/Train H', test_losses[:,9].mean(), epoch)
        sw.add_scalar('Prediction/Train G', test_losses[:,10].mean(), epoch)
        sw.add_scalar('Prediction/Train c_v', test_losses[:,11].mean(), epoch)
        sw.add_scalar('Prediction/Test μ', train_losses[:,0].mean(), epoch)
        sw.add_scalar('Prediction/Test α', train_losses[:,1].mean(), epoch)
        sw.add_scalar('Prediction/Test ε_HOMO', test_losses[:,2].mean(), epoch)
        sw.add_scalar('Prediction/Test ε_LUMO', test_losses[:,3].mean(), epoch)
        sw.add_scalar('Prediction/Test Δε', test_losses[:,4].mean(), epoch)
        sw.add_scalar('Prediction/Test <R>²', test_losses[:,5].mean(), epoch)
        sw.add_scalar('Prediction/Test ZPVE²', test_losses[:,6].mean(), epoch)
        sw.add_scalar('Prediction/Test U_0', test_losses[:,7].mean(), epoch)
        sw.add_scalar('Prediction/Test U', test_losses[:,8].mean(), epoch)
        sw.add_scalar('Prediction/Test H', test_losses[:,9].mean(), epoch)
        sw.add_scalar('Prediction/Test G', test_losses[:,10].mean(), epoch)
        sw.add_scalar('Prediction/Test c_v', test_losses[:,11].mean(), epoch)

        if (epoch+1) % args.ckpt_step == 0:
            th.save(pred_model.state_dict(), f'logs/{args.running_id}_predict/epoch_%d.th' % epoch)


def main():
    args = parser.parse_args()
    args.running_id = '%s_%s' % (args.dataset.split('.')[0], datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M'))
    if args.pretrain:
        train(args)
    if args.test:
        pred(args)


if __name__ == '__main__':
    main()