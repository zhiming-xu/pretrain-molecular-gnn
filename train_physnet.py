import torch as th
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import os
import json
from tqdm import tqdm
from datetime import datetime

from data_utils import QM7Dataset
from nn import PhysNetPretrain, PropertyPrediction


parser = ArgumentParser('PhysNet')
parser.add_argument('-data_dir', type=str, default='data')
parser.add_argument('-dataset', type=str, default='qm7_.mat')
parser.add_argument('-train_batch_size', type=int, default=1)
parser.add_argument('-ckpt_step', type=int, default=5)
parser.add_argument('-ckpt_file', type=str)
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
    dataset = QM7Dataset(data_file)
    # dataloader = DataLoader(dataset, args.train_batch_size)
    model = PhysNetPretrain()
    optim = SGD(model.parameters(), lr=args.lr)
    if args.cuda:
        model = model.cuda()
        # dataset = dataset.cuda()

    for epoch in tqdm(range(args.num_epochs)):
        loss_types, loss_poss, loss_vaes, losses = [], [], [], []
        for data in dataset:
            model.zero_grad()
            Z, R, idx_i, idx_j = data['Z'], data['R'], data['idx_i'], data['idx_j']
            if args.cuda:
                Z, R = Z.cuda(), R.cuda()
            loss_type, loss_pos, loss_vae = model(Z, R, idx_i, idx_j)

            loss = .5 * loss_pos + loss_type + .01 * loss_vae
            loss.backward()
            optim.step()

            loss_types.append(loss_type.detach().cpu().numpy())
            loss_poss.append(loss_pos.detach().cpu().numpy())
            loss_vaes.append(loss_vae.detach().cpu().numpy())
            losses.append(loss.detach().cpu().numpy())
        
        total = len(dataset)
        sw.add_scalar('Loss/Atom Type', sum(loss_types)/total, epoch)
        sw.add_scalar('Loss/Atom Position', sum(loss_poss)/total, epoch)
        sw.add_scalar('Loss/KLD', sum(loss_vaes)/total, epoch)
        sw.add_scalar('Loss/Total', sum(losses)/total, epoch)

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