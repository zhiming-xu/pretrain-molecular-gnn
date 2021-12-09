import torch as th
from torch.nn.modules import loss
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
import os.path
import json
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm

from model import PropertyPrediction, SSLMolecule, DistanceExpansion
from data_utils import *


parser = ArgumentParser()
parser.add_argument('-data_dir', type=str, default='data/')
parser.add_argument('-dataset', type=str, default='qm7')
parser.add_argument('-num_atom_types', type=int, default=5)
parser.add_argument('-atom_emb_size', type=int, default=32)
parser.add_argument('-atom_pos_size', type=int, default=3)
parser.add_argument('-cutoff', type=float, default=5.)
parser.add_argument('-dist_exp_size', type=int, default=16)
parser.add_argument('-hidden_size', type=int, default=32)
parser.add_argument('-pos_size', type=int, default=3)
parser.add_argument('-gaussian_size', type=int, default=32)
parser.add_argument('-num_type_layers', type=int, default=3)
parser.add_argument('-num_pos_layers', type=int, default=3)
parser.add_argument('-num_vae_layers', type=int, default=3)
parser.add_argument('-num_pred_layers', type=int, default=3)
parser.add_argument('-weight_l1', type=float, default=1)
parser.add_argument('-weight_l2', type=float, default=0.1)
parser.add_argument('-cuda', action='store_true')
parser.add_argument('-pretrain', action='store_true')
parser.add_argument('-test', action='store_true')
parser.add_argument('-predict_train_ratio', type=float, default=.8)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-pretrain_epochs', type=int, default=100)
parser.add_argument('-pred_epochs', type=int, default=200)
parser.add_argument('-ckpt_step', type=int, default=5)
parser.add_argument('-ckpt_file', type=str)


def main():
    args = parser.parse_args()
    args.running_id = '%s_%s' % (args.dataset, datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M'))
    # create tensorboard summary writer

    if args.pretrain:
        pretrain(args)
    if args.test:
        pred(args)


def pretrain(args):
    # create summary writer
    sw = SummaryWriter(f'logs/{args.running_id}_pretrain')
    # save arguments
    with open(f'logs/{args.running_id}_pretrain/args.json', 'w') as f:
        json.dump(vars(args), f)
    # save arguments
    datafile = os.path.join(args.data_dir, args.dataset)

    Rs, Zs, Ds, _ = load_qm7_dataset(datafile)

    model = SSLMolecule(num_atom_types=args.num_atom_types,
        atom_emb_size=args.atom_emb_size, dist_exp_size=args.dist_exp_size,
        hidden_size=args.hidden_size, pos_size=args.pos_size,
        gaussian_size=args.gaussian_size, num_type_layers=args.num_type_layers,
        num_pos_layers=args.num_pos_layers, num_vae_layers=args.num_vae_layers
    )

    if args.cuda:
        model = model.cuda()

    optimizer = SGD(model.parameters(), lr=args.lr)

    dist_exp = DistanceExpansion(repeat=args.dist_exp_size)
    cutoff = args.cutoff

    weight_l1, weight_l2 = args.weight_l1, args.weight_l2

    for epoch in tqdm(range(args.pretrain_epochs)):
        loss_atom_preds, loss_pos_preds, loss_vaes, losses = [], [], [], []
        for R, Z, D in zip(Rs, Zs, Ds):
            model.zero_grad()
            A = th.FloatTensor((D<=cutoff))
            R, Z, D = th.FloatTensor(R), th.LongTensor(Z), th.FloatTensor(D)
            if args.cuda:
                R = R.cuda(); A = A.cuda()
                D = D.cuda(); Z = Z.cuda()
            D_exp = dist_exp(D)
 
            loss_atom_pred, loss_pos_pred, loss_vae = model(R, A, D_exp, Z)
            '''
            loss = weight_l1 * (loss_atom_pred + loss_pos_pred/5) + \
                   weight_l2 * (epoch//20) * loss_vae
            # temporary workout to gradually increase kl loss weight
            if loss_atom_pred > 10 or loss_pos_pred > 120:
                continue
            '''
            loss = loss_pos_pred
            loss.backward()
            optimizer.step()
            
            loss_atom_preds.append(loss_atom_pred.detach().cpu().numpy())
            loss_pos_preds.append(loss_pos_pred.detach().cpu().numpy())
            loss_vaes.append(loss_vae.detach().cpu().numpy())
            losses.append(loss.detach().cpu().numpy())
        
        total = len(Rs)
        sw.add_scalar('Loss/Atom Type', sum(loss_atom_preds)/total, epoch)
        sw.add_scalar('Loss/Atom Position', sum(loss_pos_preds)/total, epoch)
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
    datafile = os.path.join(args.data_dir, args.dataset)

    Rs, Zs, _, Ts = load_qm7_dataset(datafile)

    pretrain_model = SSLMolecule(num_atom_types=args.num_atom_types,
        atom_emb_size=args.atom_emb_size, dist_exp_size=args.dist_exp_size,
        hidden_size=args.hidden_size, pos_size=args.pos_size,
        gaussian_size=args.gaussian_size, num_type_layers=args.num_type_layers,
        num_pos_layers=args.num_pos_layers, num_vae_layers=args.num_vae_layers
    )

    pretrain_model.load_state_dict(th.load(args.ckpt_file))

    input_size = args.atom_emb_size + args.atom_pos_size
    pred_model = PropertyPrediction(input_size, args.hidden_size, args.num_pred_layers)

    optimizer1 = SGD(pretrain_model.parameters(), lr=1e-6)
    optimizer2 = SGD(pred_model.parameters(), lr=1e-3)

    total = len(Ts)
    num_train = round(total * args.predict_train_ratio)
    num_test = total - num_train
    for epoch in tqdm(range(args.pred_epochs)):
        pred_losses, test_losses = [], []
        cnt = 0
        for R, Z, T in zip(Rs, Zs, Ts):
            if cnt < num_train:
                pretrain_model.zero_grad()
                pred_model.zero_grad()
                R, Z, T = th.FloatTensor(R), th.LongTensor(Z), th.FloatTensor(T)
                h = pretrain_model.encode(R, Z)
                loss = pred_model(h, T)
                loss.backward()
                pred_losses.append(loss)
                optimizer1.step()
                optimizer2.step()
            else:
                with th.no_grad():
                    R, Z, T = th.FloatTensor(R), th.LongTensor(Z), th.FloatTensor(T)
                    h = pretrain_model.encode(R, Z)
                    loss = pred_model(h, T)
                    test_losses.append(loss)
            cnt += 1
        
        sw.add_scalar('Loss/Pred', sum(pred_losses)/num_train, epoch)
        sw.add_scalar('Loss/Test', sum(test_losses)/num_test, epoch)


if __name__ == '__main__':
    main()