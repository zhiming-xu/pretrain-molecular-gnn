import torch as th
from torch.optim import SGD, Adam
from torch.utils.tensorboard import SummaryWriter
import os.path
import json
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm

from nn import SSLMolecule, DistanceExpansion
from data_utils import *


parser = ArgumentParser()
parser.add_argument('-data_dir', type=str, default='data/')
parser.add_argument('-dataset', type=str, default='qm7')
parser.add_argument('-num_atom_types', type=int, default=5)
parser.add_argument('-atom_emb_size', type=int, default=32)
parser.add_argument('-cutoff', type=float, default=5.)
parser.add_argument('-dist_exp_size', type=int, default=16)
parser.add_argument('-hidden_size', type=int, default=32)
parser.add_argument('-pos_size', type=int, default=3)
parser.add_argument('-gaussian_size', type=int, default=32)
parser.add_argument('-num_type_layers', type=int, default=3)
parser.add_argument('-num_pos_layers', type=int, default=3)
parser.add_argument('-num_vae_layers', type=int, default=3)
parser.add_argument('-weight_l1', type=float, default=1)
parser.add_argument('-weight_l2', type=float, default=0.1)
parser.add_argument('-cuda', action='store_true')
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-num_epochs', type=int, default=100)
parser.add_argument('-ckpt_step', type=int, default=5)


def main():
    args = parser.parse_args()
    datafile = os.path.join(args.data_dir, args.dataset)

    Rs, Zs, Ds = load_qm7(datafile)

    model = SSLMolecule(num_atom_types=args.num_atom_types,
        atom_emb_size=args.atom_emb_size, dist_exp_size=args.dist_exp_size,
        hidden_size=args.hidden_size, pos_size=args.pos_size,
        gaussian_size=args.gaussian_size, num_type_layers=args.num_type_layers,
        num_pos_layers=args.num_pos_layers, num_vae_layers=args.num_vae_layers
    )

    if args.cuda:
        model = model.cuda()

    optimizer = SGD(model.parameters(), lr=args.lr)
    running_id = '%s_%s' % (args.dataset, datetime.strftime(datetime.now(), '%Y-%m-%d_%H%M'))
    # create tensorboard summary writer
    sw = SummaryWriter(f'logs/{running_id}')
    # save arguments
    with open(f'logs/{running_id}/args.json') as f:
        json.dump(vars(args), f)

    dist_exp = DistanceExpansion(repeat=args.dist_exp_size)
    cutoff = args.cutoff

    weight_l1, weight_l2 = args.weight_l1, args.weight_l2

    for epoch in tqdm(range(args.num_epochs)):
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
            loss = weight_l1 * (loss_atom_pred + loss_pos_pred) + weight_l2 * loss_vae
            
            if loss_atom_pred > 10 or loss_pos_pred > 120:
                continue

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
            th.save(model.state_dict(), 'ckpt/epoch_%d.th' % epoch)


if __name__ == '__main__':
    main()