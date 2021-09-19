import torch as th
import os.path
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('-data_dir', type=str, default='data/')
parser.add_argument('-dataset', type=str, required=True)
parser.add_argument('-atom_emb_size', type=int, default=32)
parser.add_argument('-dist_exp_size', type=int, default=32)
parser.add_argument('-hidden_size', type=int, default=32)
parser.add_argument('-gaussian_size', type=int, default=32)
parser.add_argument('-num_type_layers', type=int, default=3)
parser.add_argument('-num_pos_layers', type=int, default=3)
parser.add_argument('-num_vae_layers', type=int, default=3)


def main():
    args = parser.parse_args()
    datafile = os.path.join(args.data_dir, args.dataset)