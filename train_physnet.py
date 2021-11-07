import torch as th
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import os
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
        dataset = DataLoader(qm9, batch_size=args.pretrain_batch_size)
    # dataloader = DataLoader(dataset, args.train_batch_size)
    model = PhysNetPretrain()
    optim = SGD(model.parameters(), lr=args.lr)
    if args.cuda:
        model = model.cuda()
        # dataset = dataset.cuda()

    for epoch in tqdm(range(args.num_epochs)):
        loss_atom_types, loss_bond_types, loss_bond_lengths, loss_bond_angles, \
            loss_length_klds, loss_angle_klds, total_losses = [], [], [], [], [], [], []
        for batch_idx, data in enumerate(tqdm(dataset, leave=False)):
            model.zero_grad()
            Z, R, idx_ijk, bonds, bond_type, bond_length, bond_angle = data.z, data.pos, data.idx_ijk, \
                data.edge_index, data.bond_type, data.bond_length, data.bond_angle
            
            loss_atom_type, loss_bond_type, \
            loss_bond_length, loss_bond_angle, \
            loss_length_kld, loss_angle_kld = model(Z, R, idx_ijk, bonds, bond_type, bond_length, bond_angle)
            # FIXME VAE loss learning schedule
            
            total_loss = loss_bond_type + loss_atom_type + loss_bond_length * 0.5 + loss_bond_angle + \
                   ((epoch//10+1)*0.07) * (loss_length_kld + loss_angle_kld)
            total_loss.backward()
            optim.step()

            loss_atom_types.append(loss_atom_type.detach().cpu().numpy())
            loss_bond_types.append(loss_bond_type.detach().cpu().numpy())
            loss_bond_lengths.append(loss_bond_length.detach().cpu().numpy())
            loss_bond_angles.append(loss_bond_angle.detach().cpu().numpy())
            loss_length_klds.append(loss_length_kld.detach().cpu().numpy())
            loss_angle_klds.append(loss_angle_kld.detach().cpu().numpy())
            total_losses.append(total_loss.detach().cpu().numpy())
            # for DEBUG
            sw.add_scalar('Debug/Atom Type', loss_atom_type, batch_idx)
            sw.add_scalar('Debug/Bond Type', loss_bond_type, batch_idx)
            sw.add_scalar('Debug/Bond Length', loss_bond_length, batch_idx)
            sw.add_scalar('Debug/Bond Angle', loss_bond_angle, batch_idx)
            sw.add_scalar('Debug/Length KLD', loss_length_kld, batch_idx)
            sw.add_scalar('Debug/Angle KLD', loss_angle_kld, batch_idx)
            sw.add_scalar('Debug/Total', total_loss, batch_idx)
 
        total = len(dataset)
        sw.add_scalar('Loss/Atom Type', sum(loss_atom_types)/total, epoch)
        sw.add_scalar('Loss/Bond Type', sum(loss_bond_types)/total, epoch)
        sw.add_scalar('Loss/Bond Length', sum(loss_bond_lengths)/total, epoch)
        sw.add_scalar('Loss/Bond Angle', sum(loss_bond_angles)/total, epoch)
        sw.add_scalar('Loss/Length KLD', sum(loss_length_klds)/total, epoch)
        sw.add_scalar('Loss/Angle KLD', sum(loss_angle_klds)/total, epoch)
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
        qm9.data.y = (qm9.data.y-qm9.data.y.mean(dim=0)) / qm9.data.y.std(dim=0)
        dataset = DataLoader(qm9, batch_size=args.pred_batch_size)
    # dataloader = DataLoader(dataset, args.train_batch_size)
    pretrain_model = PhysNetPretrain()

    pretrain_model.load_state_dict(th.load(args.ckpt_file))

    input_size = 128
    pred_model = PropertyPrediction(input_size, args.hidden_size, args.num_pred_layers)

    if args.cuda:
        pretrain_model = pretrain_model.cuda()
        pred_model = pred_model.cuda()

    optimizer1 = SGD(pretrain_model.parameters(), lr=1e-6)
    optimizer2 = SGD(pred_model.parameters(), lr=1e-3)

    total = len(dataset)
    num_train = round(total * args.pred_train_ratio)
    num_test = total - num_train
    for epoch in tqdm(range(args.pred_epochs)):
        train_losses, test_losses = [], []
        cnt = 0
        for data in tqdm(dataset, leave=False):
            Z, R, bonds = data.z, data.pos, data.edge_index
            molecule_idx = th.cat([th.zeros(data.ptr[i]-data.ptr[i-1])+i-1 for i in range(1,data.ptr.shape[0])])
            molecule_idx = molecule_idx.long()

            if cnt < num_train:
                pretrain_model.zero_grad()
                pred_model.zero_grad()
                h = pretrain_model.encode(Z, R, bonds)
                loss = pred_model(h, molecule_idx, data.y)
                loss.mean().backward()
                optimizer1.step()
                optimizer2.step()
                train_losses.append(loss.detach().cpu().mean(dim=0))
            else:
                with th.no_grad():
                    h = pretrain_model.encode(Z, R, bonds)
                    loss = pred_model(h, molecule_idx, data.y)
                    test_losses.append(loss.detach().cpu().mean(dim=0))
            cnt += 1

        train_losses = th.stack(train_losses)
        test_losses = th.stack(test_losses)
        for i in range(train_losses.shape[-1]):
            sw.add_scalar('Prediction/Train target #%s' % i, train_losses[:,i].mean(), epoch)
            sw.add_scalar('Prediction/Test target #%s' % i, test_losses[:,i].mean(), epoch)


def main():
    args = parser.parse_args()
    args.running_id = '%s_%s' % (args.dataset.split('.')[0], datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M'))
    if args.pretrain:
        train(args)
    if args.test:
        pred(args)


if __name__ == '__main__':
    main()