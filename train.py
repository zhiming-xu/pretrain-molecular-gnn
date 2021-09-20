import torch as th
from torch.optim import SGD, Adam
import os.path
from argparse import ArgumentParser

from nn import SSL_Molecule
from data_utils import *


parser = ArgumentParser()
parser.add_argument('-data_dir', type=str, default='data/')
parser.add_argument('-dataset', type=str, required=True)
parser.add_argument('-atom_emb_size', type=int, default=32)
parser.add_argument('-dist_exp_size', type=int, default=32)
parser.add_argument('-hidden_size', type=int, default=32)
parser.add_argument('-pos_size', type=int, default=3)
parser.add_argument('-gaussian_size', type=int, default=32)
parser.add_argument('-num_type_layers', type=int, default=3)
parser.add_argument('-num_pos_layers', type=int, default=3)
parser.add_argument('-num_vae_layers', type=int, default=3)
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-num_epochs', type=int, default=100)


def main():
    args = parser.parse_args()
    datafile = os.path.join(args.data_dir, args.dataset)

    Rs, Zs, Ds = load_qm9(datafile)

    model = SSL_Molecule(
        atom_emb_size=args.atom_emb_size, dist_exp_size=args.dist_exp_size,
        hidden_size=args.hidden_size, pos_size=args.pos_size,
        gaussian_size=args.gaussian_size, num_type_layers=args.num_type_layers,
        num_pos_layers=args.num_pos_layers, num_vae_layers=args.num_vae_layers
    )

    optimizer = SGD(model.parameters(), lr=args.lr)

    for epoch in range(args.num_epochs):
        for R, Z, D in zip(Rs, Zs, Ds):
            A = D[D<=5.]
            atom_type_pred, atom_pos_pred = model(R, D, A, Z)


