import torch as th
from torch.optim import SGD, Adam
import os.path
import dgl
import scipy.sparse as sp
from argparse import ArgumentParser

from nn import SSLMolecule, DistanceExpansion
from data_utils import *


parser = ArgumentParser()
parser.add_argument('-data_dir', type=str, default='data/')
parser.add_argument('-dataset', type=str, default='qm7')
parser.add_argument('-num_atom_types', type=int, default=7)
parser.add_argument('-atom_emb_size', type=int, default=32)
parser.add_argument('-cutoff', type=float, default=5.)
parser.add_argument('-dist_exp_size', type=int, default=16)
parser.add_argument('-hidden_size', type=int, default=32)
parser.add_argument('-pos_size', type=int, default=3)
parser.add_argument('-gaussian_size', type=int, default=32)
parser.add_argument('-num_type_layers', type=int, default=3)
parser.add_argument('-num_pos_layers', type=int, default=3)
parser.add_argument('-num_vae_layers', type=int, default=3)
parser.add_argument('-weight_l1', type=float, default=0.5)
parser.add_argument('-weight_l2', type=float, default=1)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-num_epochs', type=int, default=100)


def main():
    args = parser.parse_args()
    datafile = os.path.join(args.data_dir, args.dataset)

    Rs, Zs, Ds = load_qm9(datafile)

    model = SSLMolecule(num_atom_types=args.num_atom_types,
        atom_emb_size=args.atom_emb_size, dist_exp_size=args.dist_exp_size,
        hidden_size=args.hidden_size, pos_size=args.pos_size,
        gaussian_size=args.gaussian_size, num_type_layers=args.num_type_layers,
        num_pos_layers=args.num_pos_layers, num_vae_layers=args.num_vae_layers
    )

    optimizer = SGD(model.parameters(), lr=args.lr)

    dist_exp = DistanceExpansion()
    cutoff = args.cutoff

    weight_l1, weight_l2 = args.weight_l1, args.weight_l2

    for epoch in range(args.num_epochs):
        model.zero_grad()
        for R, Z, D in zip(Rs, Zs, Ds):
            A = (D <= cutoff)
            G_dist = dgl.from_scipy(sp.csr_matrix(A))
            R, Z, D = th.FloatTensor(R), th.LongTensor(Z), th.FloatTensor(D)
            D_exp = dist_exp(D)
 
            loss_atom_pred, loss_pos_pred, loss_vae = model(R, G_dist, D_exp, Z)
            loss = weight_l1 * (loss_atom_pred.sum()) + \
                   weight_l2 * (loss_pos_pred.sum() + loss_vae.sum())

            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    main()