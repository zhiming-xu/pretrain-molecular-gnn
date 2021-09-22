import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as dglfn


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
        with dist_adj.local_scope():
            feat_src = th.eisum('nf,fhk,nh->nk', dist_exp, self.bilinear_w, atom_emb)
            dist_adj.srcdata['h'] = feat_src
            dist_adj.update_all(dglfn.copy_src('h', 'm'), dglfn.sum(msg='m', out='h'))
            # FIXME: better way to exclude center node it self (masking)
            dist_adj.dstdata['h'] -= feat_src
            rst = self.activation(dist_adj.dstdata['h']) + self.bilinear_b
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
            self.layers.append(dglnn.GraphConv(in_feat_size, hidden_size, activation=F.softplus))
    
    def forward(self, atom_pos, dist_adj, atom_emb):
        feat = th.concat(atom_pos, atom_emb, dim=-1)
        for layer in self.layers:
            feat = layer(dist_adj, feat)
        return feat
        

class VAE(nn.Module):
    def __init__(self, emb_size, hidden_size, num_layers):
        '''Module for variational autoencoder'''
        super(VAE, self).__init__()
        nn_mean, nn_std = [], []
        self.kl_div = lambda mean, logstd: 0.5 * th.sum(1 + logstd - mean.square() - logstd.exp())
        for i in range(num_layers):
            if i == 0:
                nn_mean.append(nn.Linear(emb_size, hidden_size))
                nn_std.append(nn.Linear(emb_size, hidden_size))
            else:
                nn_mean.append(nn.Linear(hidden_size, hidden_size))
                nn_std.append(nn.Linear(emb_size, hidden_size))
            nn_mean.append(nn.LeakyReLU())
            nn_std.append(nn.LeakyReLU())
        
        self.nn_mean = nn.Sequential(*nn_mean)
        self.nn_logstd = nn.Sequential(*nn_std)

    def forward(self, gaussians, h):
        mean = self.nn_mean(h)
        logstd = self.nn_std(h)
        return mean + th.exp(0.5 * logstd), self.kl_div(mean, logstd)


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
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # for pretraining task 2 - central atom position prediction based on
        # surrounding atom position and embedding
        self.atom_pos_gnn = AtomPosGNN(num_pos_layers, hidden_size, atom_emb_size)
        self.atom_pos_vae = VAE(hidden_size, gaussian_size, num_vae_layers)
        self.atom_pos_linear = nn.Linear(hidden_size, pos_size)
        self.mse_loss = nn.MSELoss()

    def forward(self, atom_pos, dist_graph, dist_exp, atom_embs):
        '''
        Rs - atom position
        Zs - atom type
        '''
        num_atoms = atom_pos.shape[0]
        atom_embs = self.atom_emb(atom_embs)
        # for atom type prediction
        atom_emb_type = self.atom_type_gnn(dist_graph, dist_exp, atom_embs)
        atom_type_pred = self.atom_classifier(atom_emb_type)
        loss_atom_pred = self.bce_loss(atom_embs, atom_type_pred)

        # for atom position prediction
        atom_emb_pos = self.atom_pos_gnn(dist_graph, dist_exp, atom_embs)
        gaussians = th.rand((num_atoms, self.gaussian_size))
        atom_pos_vae, loss_vae = self.atom_pos_vae(gaussians, atom_emb_pos)
        atom_pos_pred = self.atom_pos_linear(atom_pos_vae)
        loss_pos_pred = self.mse_loss(atom_pos, atom_pos_pred)

        return loss_atom_pred, loss_pos_pred, loss_vae


class DistanceExpansion(nn.Module):
    def __init__(self, mean=0, std=.5, repeat=10):
        super(DistanceExpansion, self).__init__()
        self.gaussians = []
        for i in range(repeat):
            self.gaussians.append(
                lambda x: th.exp(-0.5*((x-(mean+i*std))/std)**2)
            )

    def forward(self, D):
        '''compute Gaussian distance expansion, R should be a vector of R^{n*1}'''
        gaussians = [g(D) for g in self.gaussians]
        return th.stack(gaussians, axis=-1)

        