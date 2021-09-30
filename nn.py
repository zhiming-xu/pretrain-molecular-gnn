import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import scipy.sparse as sp


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