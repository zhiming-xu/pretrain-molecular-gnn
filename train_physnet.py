import torch as th
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
from datetime import datetime
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from collections import defaultdict

from data_utils import QM7Dataset
from nn import PhysNetPretrain, PropertyPrediction


parser = ArgumentParser('PhysNet')
parser.add_argument('-data_dir', type=str, default='data')
parser.add_argument('-dataset', type=str, default='qm9')
parser.add_argument('-train_batch_size', type=int, default=1)
parser.add_argument('-ckpt_step', type=int, default=1)
parser.add_argument('-ckpt_file', type=str)
parser.add_argument('-pretrain_size', type=int, default=30000)
parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-cuda', action='store_true')
parser.add_argument('-pretrain', action='store_true')
parser.add_argument('-test', action='store_true')
parser.add_argument('-predict_train_ratio', type=float, default=.8)
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
        qm9 = QM9(args.data_dir)
        idx = th.randperm(len(qm9))[:args.pretrain_size]
        dataset = DataLoader(qm9[idx])
    # dataloader = DataLoader(dataset, args.train_batch_size)
    model = PhysNetPretrain()
    optim = SGD(model.parameters(), lr=args.lr)
    if args.cuda:
        model = model.cuda()
        # dataset = dataset.cuda()

    for epoch in tqdm(range(args.num_epochs)):
        loss_atom_types, loss_bond_types, loss_bond_lengths, loss_bond_angles, \
            loss_length_klds, loss_angle_klds, total_losses = [], [], [], [], [], [], []
        for sample_idx, data in tqdm(enumerate(dataset), leave=False):
            model.zero_grad()
            Z, R, bonds, bond_types = data.z, data.pos, data.edge_index, data.edge_attr
            num_atoms = Z.shape[0]
            # create idx_i and idx_j both of shape (n_atoms *(n_atoms-1))
            idx_i = th.zeros(size=(num_atoms, num_atoms-1))
            for i in range(num_atoms): idx_i[i] = i
            idx_j = th.arange(num_atoms).repeat((num_atoms, 1))
            for i in range(num_atoms): idx_j[i][i] = idx_j[i][-1]
            idx_j = idx_j[:,:-1]
            idx_i, idx_j = idx_i.long(), idx_j.long()
            # create ijk, where j is the central atom, and j,k are those bonded to it
            # j!= k, so they will form a bond angle
            ij = bonds.transpose(1, 0)
            edge_dict = defaultdict(list)
            for u, v in ij:
                edge_dict[u.item()].append(v.item())

            ijk = set()
            for u, u_bonds in edge_dict.items():
                if len(u_bonds) > 1:
                    num_u_bonds = len(u_bonds)
                    for u_i in range(num_u_bonds):
                        for u_j in range(u_i+1, num_u_bonds):
                            # the major atom studies in the middle
                            ijk.add((u_bonds[u_i], u, u_bonds[u_j]))
            idx_ijk = th.LongTensor(list(ijk))
            # use full to ensure invariance
            bonds = bonds.transpose(1, 0)
            # from one-hot to class label {0,1,2,3}
            bond_types = th.argmax(bond_types, dim=-1)
            
            if args.cuda:
                Z, R, bond_types = Z.cuda(), R.cuda(), bond_types.cuda()
 
            loss_atom_type, loss_bond_type, \
            loss_bond_length, loss_bond_angle, \
            loss_length_kld, loss_angle_kld = model(Z, R, idx_i, idx_j, idx_ijk, bonds, bond_types)
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
            sw.add_scalar('Debug/Atom Type', loss_atom_type, sample_idx)
            sw.add_scalar('Debug/Bond Type', loss_bond_type, sample_idx)
            sw.add_scalar('Debug/Bond Length', loss_bond_length, sample_idx)
            sw.add_scalar('Debug/Bond Angle', loss_bond_angle, sample_idx)
            sw.add_scalar('Debug/Length KLD', loss_length_kld, sample_idx)
            sw.add_scalar('Debug/Angle KLD', loss_angle_kld, sample_idx)
            sw.add_scalar('Debug/Total', total_loss, sample_idx)

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
    dataset = QM7Dataset(data_file)
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
    num_train = round(total * args.predict_train_ratio)
    num_test = total - num_train
    E_mean = dataset.E.mean()
    E_std = dataset.E.std()
    for epoch in tqdm(range(args.pred_epochs)):
        pred_losses, test_losses = [], []
        cnt = 0
        for data in tqdm(dataset, leave=False):
            Z, R, idx_i, idx_j, E = data['Z'], data['R'], data['idx_i'], data['idx_j'], data['E']
            E_normalized = (E - E_mean) / E_std
            if cnt < num_train:
                pretrain_model.zero_grad()
                pred_model.zero_grad()
                if args.cuda:
                    Z, R, E_normalized = Z.cuda(), R.cuda(), E_normalized.cuda()
                h = pretrain_model.encode(Z, R, idx_i, idx_j)
                loss = pred_model(h, E_normalized)
                loss.backward()
                pred_losses.append(loss)
                # optimizer1.step()
                optimizer2.step()
            else:
                with th.no_grad():
                    if args.cuda:
                        Z, R, E_normalized = Z.cuda(), R.cuda(), E_normalized.cuda()
                    h = pretrain_model.encode(Z, R, idx_i, idx_j)
                    loss = pred_model(h, E_normalized)
                    test_losses.append(loss)
            cnt += 1
        
        sw.add_scalar('Loss/Pred', sum(pred_losses)/num_train, epoch)
        sw.add_scalar('Loss/Test', sum(test_losses)/num_test, epoch)


def main():
    args = parser.parse_args()
    args.running_id = '%s_%s' % (args.dataset.split('.')[0], datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M'))
    if args.pretrain:
        train(args)
    if args.test:
        pred(args)


if __name__ == '__main__':
    main()