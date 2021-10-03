import torch as th
from torch._C import namedtuple_eigenvalues_eigenvectors
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import scipy.sparse as sp
from torch.nn.modules import activation


class AtomEmbedding(nn.Module):
    ''''Module for atom embedding'''
    def __init__(self, num_atom_types, emb_size):
        super(AtomEmbedding, self).__init__()
        self.atom_emb = nn.Embedding(num_atom_types, emb_size)

    def forward(self, x):
        return self.atom_emb(x)


class AtomTypeGNN(nn.Module):
    def __init__(self, dist_exp_size, atom_emb_size, hidden_size):
        '''Module for message-passing in atom type prediction task'''
        super(AtomTypeGNN, self).__init__()
        bilinear_w = nn.Parameter(th.FloatTensor(
            th.rand(dist_exp_size, atom_emb_size, hidden_size)
        ))
        bilinear_b = nn.Parameter(th.FloatTensor(
            th.rand(hidden_size)
        ))
        self.activation = nn.Softplus()
        self.register_parameter('bilinear_w', bilinear_w)
        self.register_parameter('bilinear_b', bilinear_b)

    def forward(self, dist_adj, dist_exp, atom_emb):
        adj_exp = th.einsum('mn,mnk->mk', dist_adj, dist_exp)
        feat = th.einsum('nf,fhk,nh->nk', adj_exp, self.bilinear_w, atom_emb)
        rst = self.activation(feat) + self.bilinear_b
        return rst


class AtomTypeClassifier(nn.Module):
    '''Module for predicting central atom type based on neighbor information'''
    def __init__(self, num_layers, emb_size, hidden_size, num_types):
        super(AtomTypeClassifier, self).__init__()
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

    
class AtomPosGNN(nn.Module):
    def __init__(self, num_layers, hidden_size, atom_emb_size, pos_size=3):
        '''Module for message-passing in position prediction task'''
        super(AtomPosGNN, self).__init__()
        self.layers = nn.ModuleList()
        in_feat_size = atom_emb_size + pos_size # concat atom embedding and position
        for i in range(num_layers):
            # should be no zero indegree
            if i == 0:
                self.layers.append(dglnn.GraphConv(in_feat_size, hidden_size, activation=F.softplus))
            else:
                self.layers.append(dglnn.GraphConv(hidden_size, hidden_size, activation=F.softplus))
    
    def forward(self, atom_pos, dist_adj, atom_emb):
        dist_adj -= th.eye(dist_adj.shape[0])
        graph = dgl.from_scipy(sp.csr_matrix(dist_adj))
        feat = th.cat([atom_emb, atom_pos], dim=-1)
        for layer in self.layers:
            feat = layer(graph, feat)
        return feat
        

class VAE(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers):
        '''Module for variational autoencoder'''
        super(VAE, self).__init__()
        nn_mean, nn_logstd = [], []
        self.kld_loss = lambda mean, logstd: -0.5 * th.sum(1 + logstd - mean.square() - logstd.exp())
        for i in range(num_layers):
            if i == 0:
                nn_mean.append(nn.Linear(emb_size, hidden_size))
                nn_logstd.append(nn.Linear(emb_size, hidden_size))
            else:
                nn_mean.append(nn.Linear(hidden_size, hidden_size))
                nn_logstd.append(nn.Linear(emb_size, hidden_size))
            nn_mean.append(nn.Softplus())
            nn_logstd.append(nn.Softplus())
        
        self.nn_mean = nn.Sequential(*nn_mean)
        self.nn_logstd = nn.Sequential(*nn_logstd)

    def forward(self, gaussians, h):
        mean = self.nn_mean(h)
        logstd = self.nn_logstd(h)
        return mean + gaussians * th.exp(0.5 * logstd), self.kld_loss(mean, logstd)


class SSLMolecule(nn.Module):
    def __init__(self, num_atom_types, atom_emb_size, dist_exp_size, hidden_size,
                 pos_size, gaussian_size, num_type_layers, num_pos_layers, num_vae_layers):
        super(SSLMolecule, self).__init__()
        assert(pos_size==3, '3D coordinates required')
        self.gaussian_size = gaussian_size
        # shared atom embedding
        self.atom_emb = AtomEmbedding(num_atom_types, atom_emb_size)
        # for pretraining task 1 - central atom type prediction based on
        # ditance expansion and atom embedding
        self.atom_type_gnn = AtomTypeGNN(dist_exp_size, atom_emb_size, hidden_size)
        self.atom_classifier = AtomTypeClassifier(num_type_layers, hidden_size, hidden_size,
                                                  num_atom_types)
        self.ce_loss = nn.CrossEntropyLoss()
        
        # for pretraining task 2 - central atom position prediction based on
        # surrounding atom position and embedding
        self.atom_pos_gnn = AtomPosGNN(num_pos_layers, hidden_size, atom_emb_size)
        self.atom_pos_vae = VAE(hidden_size, gaussian_size, num_vae_layers)
        self.atom_pos_linear = nn.Linear(hidden_size, pos_size)
        self.mse_loss = nn.MSELoss()

    def forward(self, atom_pos, dist_adj, dist_exp, atom_types):
        '''
        Rs - atom position
        Zs - atom type
        '''
        num_atoms = atom_pos.shape[0]
        atom_embs = self.atom_emb(atom_types)
        # for atom type prediction
        atom_emb_type = self.atom_type_gnn(dist_adj, dist_exp, atom_embs)
        atom_type_pred = self.atom_classifier(atom_emb_type)
        loss_atom_pred = self.ce_loss(atom_type_pred, atom_types)

        # for atom position prediction
        atom_emb_pos = self.atom_pos_gnn(atom_pos, dist_adj, atom_embs)
        gaussians = th.rand((num_atoms, self.gaussian_size))
        atom_pos_vae, loss_vae = self.atom_pos_vae(gaussians, atom_emb_pos)
        atom_pos_pred = self.atom_pos_linear(atom_pos_vae)
        loss_pos_pred = self.mse_loss(atom_pos, atom_pos_pred)

        return loss_atom_pred, loss_pos_pred, loss_vae

    def encode(self, atom_pos, atom_types):
        atom_embs = self.atom_emb(atom_types)
        return th.cat([atom_embs, atom_pos], axis=-1)


class Gaussian(nn.Module):
    def __init__(self, mean, std):
        super(Gaussian, self).__init__()
        self.mean, self.std = mean, std

    def forward(self, x):
        return th.exp(-0.5*((x-self.mean)/self.std)**2)


class DistanceExpansion(nn.Module):
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
    def __init__(self, input_size, hidden_size=32, num_layers=3):
        super(PropertyPrediction, self).__init__()
        W = nn.Parameter(th.rand(input_size))
        b = nn.Parameter(th.rand(input_size))
        self.register_parameter('W', W)
        self.register_parameter('b', b)
        self.loss = nn.L1Loss()

        nets = []
        for i in range(num_layers):
            if i == 0:
                nets.append(nn.Linear(input_size, hidden_size))
            elif i == num_layers - 1:
                nets.append(nn.Linear(hidden_size, 1))
                # break # don't apply softplus for the last layer
            else:
                nets.append(nn.Linear(hidden_size, hidden_size))
            nets.append(nn.Tanh())
        
        self.nn = nn.Sequential(*nets)
    
    def forward(self, x, target):
        h = th.einsum('nf,f->f', x, self.W) + self.b
        h = self.nn(h)
        return self.loss(h, target)


class ResidualLayer(nn.Module):
    def __init__(self, in_size, out_size, activation_fn, use_bias=True, drop_prob=.5):
        super(ResidualLayer, self).__init__()
        self._keep_prob = drop_prob
        self.activation_fn = activation_fn
        self.dense1 = nn.Linear(in_size, out_size, bias=use_bias)
        self.dense2 = nn.Linear(in_size, out_size, bias=use_bias)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        if self.activation_fn:
            y = self.dropout(self.activation_fn(x))
        else:
            y = self.dropout(x)
        x = x + self.dense2(self.dense1(y))
        return x


class PhysNetInteractionLayer(nn.Module):
    def __init__(self, K, F, num_residual, activation_fn=None, drop_prob=.5):
        super(PhysNetInteractionLayer, self).__init__()
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(drop_prob)
        self.k2f = nn.Linear(K, F, bias=False)
        self.dense_i = nn.Linear(F, F)
        self.dense_j = nn.Linear(F, F)
        self.residuals = nn.ModuleList()
        for _ in range(num_residual):
            self.residuals.append(ResidualLayer(F, F, activation_fn, drop_prob))
        self.dense = nn.Linear(F, F) 
        self.u = nn.Parameter(nn.rand(F))

    def forward(self, x, rbf, idx_i, idx_j):
        if self.activation_fn:
            xa = self.dropout(self.activation_fn(x))
        else:
            xa = self.dropout(x)
        
        g = self.k2f(rbf)
        xi = self.dense_i(xa)
        xj = th.scatter_add_(0, idx_i, g*th.gather(self.dense_j(xa), 0, idx_j))

        m = xi + xj

        for layer in self.residuals:
            m = layer(m)
        
        if self.activation_fn:
            m = self.activation_fn(m)
        
        x = self.u * x + self.dense(m)
        return x


class PhysNetRBFLayer(nn.Module):
    def __init__(self, cutoff, K):
        super(PhysNetRBFLayer, self).__init__()
        self.K = K
        self.cutoff = cutoff
        def softplus_inverse(x):
            return x + th.log(-th.exp(x)-1)
        centers = softplus_inverse(th.linspace(1., th.exp(-cutoff), K))
        self.centers = nn.functional.softplus(centers)

        widths = [softplus_inverse((.5/((1-th.exp(-cutoff))/K))**2)] * K
        self.widths = nn.functional.softplus(widths)

    def cutoff_fn(self, D):
        x = D/self.cutoff
        x3 = x**3
        x4 = x3*x
        x5 = x4*x
        return th.where(x<1, 1 - 6*x5 + 15*x4 - 10*x3, th.zeros_like(x))

    def forward(self, D):
        D = th.unsqueeze(D, -1)
        rbf = self.cutoff_fn(D) * th.exp(-self.widths*(th.exp(-D)-self.centers)**2)
        return rbf


class PhysNetInteractionBlock(nn.Module):
    def __init__(self, K, F, num_residual_atom, num_residual_interaction, activation_fn, drop_prob=.5):
        super(PhysNetInteractionBlock, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.activation_fn = activation_fn
        self.interaction = PhysNetInteractionLayer(K, F, num_residual_interaction, activation_fn, drop_prob)
        self.residuals = nn.ModuleList()
        for _ in range(num_residual_atom):
            self.residuals.append(ResidualLayer(F, F, activation_fn=activation_fn, drop_prob=drop_prob))
        
    def forward(self, x, rbf, idx_i, idx_j):
        x = self.interaction(x, rbf, idx_i, idx_j)
        for layer in self.residuals:
            x = layer(x)
        return x


class PhysNetOutputBlock(nn.Module):
    def __init__(self, F, num_residual, activation_fn, drop_prob=.5):
        super(PhysNetOutputBlock, self).__init__()
        self.activation_fn = activation_fn
        self.residuals = nn.ModuleList()
        for _ in range(num_residual):
            self.residuals.append(ResidualLayer(F, F, activation_fn=activation_fn, drop_prob=drop_prob))
        self.dense = nn.Linear(F, 2, bias=False)

    def forward(self, x):
        for layer in self.residuals:
            x = layer(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return self.dense(x)


def shifted_softplus(x):
    return nn.functional.softplus(x) - th.log(2.)


class PhysNet(nn.Module):
    def __init__(self, F, K, short_cutoff, long_cutoff=None, num_blocks=3, num_residual_atom=2,
                 num_residual_interaction=2, num_residual_output=1, drop_prob=.5, activation_fn=shifted_softplus):
        super(PhysNet, self).__init__()
        self.num_blocks = num_blocks
        self.atom_embedding = AtomEmbedding(20, F)
        self.rbf_layer = PhysNetRBFLayer(short_cutoff, K)
        self.interaction_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.interaction_blocks.append(
                PhysNetInteractionBlock(K, F, num_residual_atom, num_residual_interaction,
                activation_fn, drop_prob)
            )
            self.output_blocks.append(
                PhysNetOutputBlock(F, num_residual_output, activation_fn, drop_prob)
            )

    def calculate_interatom_distance(self, R, idx_i, idx_j, offsets=None):
        Ri = th.gather(R, idx_i)
        Rj = th.gather(R, idx_j)
        if offsets:
            R += offsets
        Dij = th.sqrt(nn.functional.relu(th.sum((Ri-Rj)**2, -1)))
        return Dij

    def forward(self, Z, R, idx_i, idx_j, offsets=None, short_idx_i=None, short_idx_j=None, short_offsets=None):
        D_ij_lr = self.calculate_interatom_distance(R, idx_i, idx_j, offsets=offsets)
        if short_idx_i is not None and short_idx_j is not None:
            D_ij_short = self.calculate_interatom_distance(R, short_idx_i, short_idx_j, offsets=offsets)
        else:
            short_idx_i = idx_i
            short_idx_j = idx_j
            D_ij_short = D_ij_lr
        
        rbf = self.rbf_layer(D_ij_short)
        x = th.gather(self.atom_embedding, Z)

        E_total, Q_total = 0, 0
        nhloss, last_out2 = 0, 0
        for i in range(self.num_blocks):
            x = self.interaction_blocks[i](x, rbf, short_idx_i, short_idx_j)
            out = self.output_blocks[i](x)
            E_total += out[:, 0]
            Q_total += out[:, 1]
            out2 = out**2
            if i > 0:
                nhloss += th.mean(out2/(out2 + last_out2 + 1e-7))
        
        return E_total, Q_total, nhloss
            