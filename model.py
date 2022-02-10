#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from typing import Union, Tuple, Optional
from torch_geometric.typing import PairTensor, Adj, OptTensor
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax
th.manual_seed(42)

from nn_utils import *


class MLPClassifier(nn.Module):
    '''Module for predicting central atom type based on neighbor information'''
    def __init__(self, num_layers, emb_size, hidden_size, num_types):
        super(MLPClassifier, self).__init__()
        classifier = []
        assert(num_layers>1, '# of total layers must be larger than 1')
        for i in range(num_layers):
            if i == 0:
                classifier.append(nn.Linear(emb_size, hidden_size))
            elif i == num_layers - 1:
                classifier.append(nn.Linear(hidden_size, num_types))
            else:
                classifier.append(nn.Linear(hidden_size, hidden_size))
            classifier.append(nn.Softplus())
        self.classifier = nn.Sequential(*classifier)

    def forward(self, h):
        logits = self.classifier(h)
        return logits


class DualEmb(nn.Module):
    def __init__(self, emb_size, hidden_size, bias=False):
        super().__init__()
        self.linear = nn.Linear(emb_size, hidden_size, bias=bias)

    def forward(self, h1, h2):
        h1 = self.linear(h1)
        h2 = self.linear(h2)
        return th.cat([h1 * h2, h1 + h2], dim=-1)


class BilinearClassifier(MLPClassifier):
    def __init__(self, num_layers, emb_size, hidden_size, num_types, bias=False):
        super(BilinearClassifier, self).__init__(num_layers-1, hidden_size*2, hidden_size, num_types)
        self.dual_emb = DualEmb(emb_size, hidden_size)

    def forward(self, h1, h2):
        h = self.dual_emb(h1, h2)
        logits = self.classifier(h)
        return logits

 
class VAE(nn.Module):
    '''Module for variational autoencoder'''
    def __init__(self, emb_size, hidden_size, num_layers):
        super().__init__()
        nn_mean, nn_logstd = [], []
        self.kld_loss = lambda mean, logstd: -0.5 * th.sum(1 + logstd - mean.square() - logstd.exp())
        for i in range(num_layers):
            if i == 0:
                nn_mean.append(nn.Linear(emb_size, hidden_size))
                nn_logstd.append(nn.Linear(emb_size, hidden_size))
            else:
                nn_mean.append(nn.Linear(hidden_size, hidden_size))
                nn_logstd.append(nn.Linear(hidden_size, hidden_size))
            nn_mean.append(nn.Softplus())
            nn_logstd.append(nn.Softplus())
 
        self.nn_mean = nn.Sequential(*nn_mean)
        self.nn_logstd = nn.Sequential(*nn_logstd)

    def forward(self, gaussians, h):
        mean = self.nn_mean(h)
        logstd = self.nn_logstd(h)
        return mean + gaussians * th.exp(0.5 * logstd), self.kld_loss(mean, logstd)


class DualVAE(VAE):
    def __init__(self, emb_size, hidden_size, num_layers, bias=False):
        super(DualVAE, self).__init__(hidden_size*2, hidden_size, num_layers-1)
        self.dual_emb = DualEmb(emb_size, hidden_size, bias=bias)

    def forward(self, gaussians, atom_repr1, atom_repr2):
        h = self.dual_emb(atom_repr1, atom_repr2)
        mean = self.nn_mean(h)
        logstd = self.nn_logstd(h)
        return mean + gaussians * th.exp(0.5 * logstd), self.kld_loss(mean, logstd)


class TriVAE(VAE):
    def __init__(self, emb_size, hidden_size, num_layers, bias=False):
        super(TriVAE, self).__init__(hidden_size*2, hidden_size, num_layers-1)
        self.dual_emb1 = DualEmb(emb_size, hidden_size, bias=bias)
        self.dual_emb2 = DualEmb(hidden_size*2, hidden_size, bias=bias)

    def forward(self, gaussians, atom_repr1, atom_repr2, atom_repr3):
        h1 = self.dual_emb1(atom_repr1, atom_repr2)
        h2 = self.dual_emb1(atom_repr2, atom_repr3)
        h = self.dual_emb2(h1, h2)
        mean = self.nn_mean(h)
        logstd = self.nn_logstd(h)
        return mean + gaussians * th.exp(0.5 * logstd), self.kld_loss(mean, logstd)


class QuadVAE(VAE):
    def __init__(self, emb_size, hidden_size, num_layers, bias=False):
        super(QuadVAE, self).__init__(hidden_size*2, hidden_size, num_layers-1)
        self.dual_emb_uv = DualEmb(emb_size, hidden_size, bias=bias)
        self.reduce = nn.Linear(emb_size, hidden_size*2, bias=bias)
        self.dual_emb_all = DualEmb(hidden_size*4, hidden_size, bias=bias)

    def forward(self, gaussians, h_l, h_u, h_v, h_k):
        h_uv = self.dual_emb_uv(h_u, h_v)
        h_l, h_k = self.reduce(h_l), self.reduce(h_k)
        h_luv = th.cat([h_uv * h_l, h_uv + h_l], dim=-1)
        h_uvk = th.cat([h_uv * h_k, h_uv + h_k], dim=-1)
        h = self.dual_emb_all(h_luv, h_uvk)
        mean = self.nn_mean(h)
        logstd = self.nn_logstd(h)
        return mean + gaussians * th.exp(0.5 * logstd), self.kld_loss(mean, logstd)


class Gaussian(nn.Module):
    '''one instance of Gaussian expansion'''
    def __init__(self, mean, std):
        super(Gaussian, self).__init__()
        self.mean, self.std = mean, std

    def forward(self, x):
        return th.exp(-0.5*((x-self.mean)/self.std)**2)


class DistanceExpansion(nn.Module):
    ''''Distance expansion module, a series of Gaussians'''
    def __init__(self, mean=0, std=1, step=0.2, repeat=10):
        super(DistanceExpansion, self).__init__()
        self.gaussians = nn.ModuleList(
            [Gaussian(mean+i*step, std) for i in range(repeat)]
        )

    def forward(self, D):
        '''compute Gaussian distance expansion, R should be a vector of R^{n*1}'''
        gaussians = [g(D) for g in self.gaussians]
        return th.stack(gaussians, axis=-1)


class PropertyPrediction(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=3, slop=.1):
        super(PropertyPrediction, self).__init__()

        nets = []
        for i in range(num_layers):
            if i == 0:
                nets.append(nn.Linear(input_size, hidden_size))
            else:
                nets.append(nn.Linear(hidden_size, hidden_size))
            nets.append(nn.LeakyReLU(slop))
        
        self.nn = nn.Sequential(*nets)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(slop),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x, batch):
        h = self.nn(x)
        pred_target = self.output_layer(scatter_add(h, batch.to(h.device), dim=0))
        return pred_target


class ExponentialRBF(nn.Module):
    def __init__(self, cutoff, rbf_size):
        super(ExponentialRBF, self).__init__()
        self.K = rbf_size
        self.cutoff = cutoff
        centers = softplus_inverse(np.linspace(1., np.exp(-cutoff), rbf_size))
        self.register_buffer('centers', F.softplus(th.FloatTensor(centers)))

        widths = th.FloatTensor([softplus_inverse((.5/((1.-np.exp(-cutoff))/rbf_size))**2)] * rbf_size)
        self.register_buffer('widths', F.softplus(widths))

    def cutoff_fn(self, D):
        x = D/self.cutoff
        x3 = x**3
        x4 = x3*x
        x5 = x4*x
        return th.where(x<1, 1 - 6*x5 + 15*x4 - 10*x3, th.zeros_like(x))

    def forward(self, D):
        if D.shape[-1] != 1:
            D = th.unsqueeze(D, -1)
        rbf = self.cutoff_fn(D) * th.exp(-self.widths*(th.exp(-D)-self.centers)**2)
        return rbf


class PMNetEncoder(nn.Module):
    def __init__(self, num_feat, hidden_size, num_head, rbf_size, num_att_layer, slop=.1, dropout=.5):
        super().__init__()
        self.num_att_layer = num_att_layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.atom_emb = nn.Linear(num_feat, hidden_size)
        self.transformers = nn.ModuleList(
            PMNetEncoderLayer(hidden_size, hidden_size//num_head, num_head, beta=True, rbf_size=rbf_size,
                              dropout=dropout)
            for _ in range(num_att_layer)
        )
        self.ffns = nn.ModuleList(nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(slop),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(slop),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(slop))
            for _ in range(num_att_layer)
        )
        self.activation = nn.LeakyReLU(slop)
        # self.ipa_transformer = IPATransformer(dim=hidden_size, depth=num_att_layer, require_pairwise_repr=False)

    def forward(self, atom_embs, edge_indices, pos, edge_weight):
        atom_embs = self.activation(self.atom_emb(atom_embs))
        Xs = []
        for i in range(self.num_att_layer):
            # add-norm for transformer layer
            atom_embs = atom_embs + self.activation(
                self.transformers[i](atom_embs, edge_indices, pos, edge_weight)
            )
            atom_embs = self.layer_norm(atom_embs)
            # add-norm for ffn layer
            atom_embs = atom_embs + self.ffns[i](atom_embs)
            atom_embs = self.layer_norm(atom_embs)
            Xs.append(atom_embs)
 
        # X = th.stack(Xs, dim=0).transpose(1, 0)
        # X = self.ipa_transformer(X)[0].transpose(1, 0).sum(dim=0)
        # X = self.layer_norm(X)
        return Xs


class PMNetDecoder(nn.Module):
    def __init__(self, num_feats, hidden_size, num_head, rbf_size, num_att_layer, slop=.1, dropout=.5):
        super().__init__()
        self.num_att_layer = num_att_layer
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.atom_emb = nn.Linear(num_feats, hidden_size)
        self.self_attentions = nn.ModuleList(
            PMNetEncoderLayer(hidden_size, hidden_size//num_head, num_head, beta=True, rbf_size=rbf_size,
                              dropout=dropout)
            for _ in range(num_att_layer)
        )
        self.cross_attentions = nn.ModuleList(
            PMNetEncoderLayer(hidden_size, hidden_size//num_head, num_head, beta=True, rbf_size=rbf_size,
                              dropout=dropout)
            for _ in range(num_att_layer)
        )
        self.ffns = nn.ModuleList(nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(slop),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(slop),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(slop))
            for _ in range(num_att_layer)
        )
        self.activation = nn.LeakyReLU(slop)
        # self.ipa_transformer = IPATransformer(dim=hidden_size, depth=num_att_layer, require_pairwise_repr=False)

    def forward(self, atom_embs_enc, atom_embs, edge_indices, pos, edge_weight):
        # atom_embs = self.atom_emb(atom_ids)
        Xs = []
        for i in range(self.num_att_layer):
            # add-norm for self-attention transformer
            atom_embs = atom_embs + self.activation(
                self.self_attentions[i](atom_embs, edge_indices, pos, edge_weight)
            )
            atom_embs = self.layer_norm(atom_embs)
            # add-norm for transformer layer
            atom_embs = atom_embs + self.activation(
                self.cross_attentions[i]((atom_embs_enc[i], atom_embs), edge_indices, pos, edge_weight)
            )
            atom_embs = self.layer_norm(atom_embs)
            # add-norm for ffn layer
            atom_embs = atom_embs + self.ffns[i](atom_embs)
            atom_embs = self.layer_norm(atom_embs)
            Xs.append(atom_embs)
 
        # X = th.stack(Xs, dim=0).transpose(1, 0)
        # X = self.ipa_transformer(X)[0].transpose(1, 0).sum(dim=0)
        # X = self.layer_norm(X)
        return Xs


class PMNetPretrainer(nn.Module):
    def __init__(self, in_size, hidden_size, num_elems, num_bond_types, slop=.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.atom_type_classifier = MLPClassifier(3, in_size, hidden_size, num_elems)
        self.bond_type_classifier = BilinearClassifier(3, in_size, hidden_size, num_bond_types)
        self.bond_length_vae = DualVAE(in_size, hidden_size, 3)
        self.bond_length_linear = nn.Sequential(nn.Linear(hidden_size, 1), nn.LeakyReLU(slop))
        self.bond_angle_vae = TriVAE(in_size, hidden_size, 3)
        self.bond_angle_linear = nn.Sequential(nn.Linear(hidden_size, 1), nn.LeakyReLU(slop))
        self.torsion_vae = QuadVAE(in_size, hidden_size, 3)
        self.torsion_linear = nn.Sequential(nn.Linear(hidden_size, 1), nn.LeakyReLU(slop))

    def forward(self, X, bonds, idx_ijk, plane):
        # predict atom type
        atom_type_pred = self.atom_type_classifier(X)
        # predict bond type
        atom_reprs = X[bonds]
        bond_type_pred = self.bond_type_classifier(atom_reprs[0,:], atom_reprs[1,:])
        # predict bond length
        gaussians = th.rand((bonds.shape[1], self.hidden_size)).to(X.device)
        bond_length_h, loss_length_kld = self.bond_length_vae(gaussians, atom_reprs[0,:], atom_reprs[1,:])
        bond_length_pred = self.bond_length_linear(bond_length_h)
        # FIXME no normalization on bond lengths is performed currently
        # predict bond angle
        atom_reprs = X[idx_ijk]
        gaussians = th.rand((idx_ijk.shape[0], self.hidden_size)).to(X.device)
        bond_angle_h, loss_angle_kld = self.bond_angle_vae(
            gaussians, atom_reprs[:,0], atom_reprs[:,1], atom_reprs[:,2]
        )
        bond_angle_pred = self.bond_angle_linear(bond_angle_h)
        # TODO check dihedral angle (torsion) implementation
        atom_reprs = X[plane]
        gaussians = th.rand((plane.shape[0], self.hidden_size)).to(X.device)
        torsion_h, loss_torsion_kld = self.torsion_vae(
            gaussians, atom_reprs[:,0], atom_reprs[:, 1], atom_reprs[:, 2], atom_reprs[:, 3]
        )
        torsion_pred = self.torsion_linear(torsion_h)
        return atom_type_pred, bond_type_pred, bond_length_pred, bond_angle_pred, torsion_pred, \
               loss_length_kld, loss_angle_kld, loss_torsion_kld


class PMNet(nn.Module):
    def __init__(self, hidden_size=64, num_head=8, rbf_size=9, num_att_layer=6,
                 num_feats=28, num_elems=5, num_bond_types=4, dropout=0.5, mode='pretrain'):
        super().__init__()
        self.encoder = PMNetEncoder(num_feats, hidden_size, num_head, rbf_size,
                                    num_att_layer, dropout=dropout)
        self.decoder = PMNetDecoder(num_feats, hidden_size, num_head, rbf_size,
                                    num_att_layer, dropout=dropout)
        if mode == 'pretrain':
            self.generator = PMNetPretrainer(hidden_size, hidden_size//4, num_elems, num_bond_types)
        elif mode == 'pred':
            self.generator = PropertyPrediction(hidden_size, hidden_size//4)
        self.mode = mode

    def forward(self, X, R, bonds, edge_weight, **args):
        Xs = self.encoder(X, bonds, R, edge_weight)
        Xs = self.decoder(Xs, Xs[-1], bonds, R, edge_weight)
        if self.mode == 'pretrain':
            return self.generator(Xs[-1], args['idx_ij'], args['idx_ijk'], args['plane'])
        elif self.mode == 'pred':
            return self.generator(Xs[-1], args['batch'])



class PMNetEncoderLayer(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        bias: bool = True,
        root_weight: bool = True,
        rbf_size: int = 9,
        cutoff: float = 10,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(PMNetEncoderLayer, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.edge_dim = rbf_size + 1
        self._alpha = None

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.rbf = ExponentialRBF(cutoff=cutoff, rbf_size=rbf_size)
        self.lin_key = Linear(in_channels[0], heads * out_channels)
        self.lin_query = Linear(in_channels[1], heads * out_channels)
        self.lin_value = Linear(in_channels[0], heads * out_channels)
        self.lin_edge = Linear(self.edge_dim, heads * out_channels, bias=False)
        self.lin_qk1 = Linear(out_channels, out_channels)
        self.lin_qk2 = Linear(3 * out_channels, out_channels)

        if concat:
            self.lin_skip = Linear(in_channels[1], heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels[1], out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()


    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, pos: OptTensor,
                edge_weight: OptTensor, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        query = self.lin_query(x[1]).view(-1, H, C)
        key = self.lin_key(x[0]).view(-1, H, C)
        value = self.lin_value(x[0]).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value, pos=pos,
                             edge_weight=edge_weight, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(th.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                pos_i: OptTensor, pos_j: OptTensor, edge_weight: Tensor,
                index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        w_pos = 0
        if pos_i is not None and pos_j is not None:
            dist = th.norm(pos_i-pos_j, dim=-1)
            dist_exp = self.rbf(dist)
            w_pos = self.lin_edge(
                th.cat([dist_exp, edge_weight.unsqueeze(-1)], dim=-1)
            ).view(-1, self.heads, self.out_channels)
            w_pos = self.dropout(w_pos)

        w_alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        w_alpha = self.dropout(softmax(w_alpha, index, ptr, size_i))
        self._alpha = w_alpha

        query_i_t = self.lin_qk1(query_i)
        key_j_t = self.lin_qk1(key_j)
        w_dir = self.lin_qk2(th.cat(
            [query_i_t+key_j_t, query_i_t-key_j_t, query_i_t*key_j_t],
            dim=-1)
        )
        w_dir = self.dropout(w_dir)

        out = value_j * (w_alpha.view(-1, self.heads, 1) + w_pos + w_dir)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


class PMNetPredictionLayer(PMNetEncoderLayer):
    def forward(self, query, key, value, edge_index: Adj, pos: OptTensor,
                edge_weight: OptTensor, return_attention_weights=None):
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """

        H, C = self.heads, self.out_channels
        q = self.lin_query(query).view(-1, H, C)
        k = self.lin_key(key).view(-1, H, C)
        v = self.lin_value(value).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=q, key=k, value=v, pos=pos,
                             edge_weight=edge_weight, size=None)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.root_weight:
            x_r = self.lin_skip(query)
            if self.lin_beta is not None:
                beta = self.lin_beta(th.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out