import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
import dgl.function as dglfn


class AtomEmbedding(nn.Module):
    ''''Module for atom embedding'''
    def __init__(self, emb_size):
        super(AtomEmbedding, self).__init__()
        self.atom_emb = nn.Embedding(emb_size)

    def forward(self, x):
        return self.atom_emb(x)


class AtomTypeGNN(nn.Module):
    def __init__(self, dist_exp_size, atom_emb_size, hidden_size):
        '''Module for message-passing in atom type prediction task'''
        super(AtomTypeGNN, self).__init__()
        self.bilinear_w = nn.Parameter(th.FloatTensor(
            size=(dist_exp_size, atom_emb_size, hidden_size)
        ))
        self.bilinear_b = nn.Parameter(th.FloatTensor(
            size=(hidden_size)
        ))
        self.activation = nn.Softplus()

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
        self.classifier = []
        assert(num_layers>1, '# of total layers must be larger than 1')
        for i in range(num_layers):
            if i == 0:
                self.classifier.append(nn.Linear(emb_size, hidden_size))
            elif i == num_layers - 1:
                self.classifier.append(nn.Linear(hidden_size, num_types))
            else:
                self.classifier.append(nn.Linear(hidden_size, hidden_size))
            self.classifier.append(nn.Softplus())
        self.classifier = nn.Sequential(self.classifier)

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
        self.nn_mean, self.nn_std = [], []
        for i in range(num_layers):
            if i == 0:
                self.nn_mean.append(nn.Linear(emb_size, hidden_size))
                self.nn_std.append(nn.Linear(emb_size, hidden_size))
            else:
                self.nn_mean.append(nn.Linear(hidden_size, hidden_size))
                self.nn_std.append(nn.Linear(emb_size, hidden_size))
            self.nn_mean.append(nn.LeakyReLU())
            self.nn_std.append(nn.LeakyReLU())
        
        self.nn_mean = nn.Sequential(self.nn_mean)
        self.nn_std = nn.Sequential(self.nn_std)

    def forward(self, gaussians, h):
        mean = self.nn_mean(h)
        std = self.nn_std(h)
        return gaussians * std + mean


class SSLMolecule(nn.Module):
    def __init__(self, atom_emb_size, dist_exp_size, hidden_size, pos_size,
                 gaussian_size, num_type_layers, num_pos_layers, num_vae_layers):
        super(SSLMolecule, self).__init__()
        assert(pos_size==3, '3D coordinates required')
        self.gaussian_size = gaussian_size
        # shared atom embedding
        self.atom_emb = AtomEmbedding(atom_emb_size)
        # for pretraining task 1 - central atom type prediction based on
        # ditance expansion and atom embedding
        self.atom_type_gnn = AtomTypeGNN(dist_exp_size, atom_emb_size, hidden_size)
        self.atom_classifier = AtomTypeClassifier(num_type_layers, hidden_size, hidden_size)
        
        # for pretraining task 2 - central atom position prediction based on
        # surrounding atom position and embedding
        self.atom_pos_gnn = AtomPosGNN(num_pos_layers, hidden_size, atom_emb_size)
        self.atom_pos_vae = VAE(hidden_size, gaussian_size, num_vae_layers)
        self.atom_pos_linear = nn.Linear(hidden_size, pos_size)

    def forward(self, dist_graph, dist_exp, atom_emb):
        '''
        Rs - atom position
        Zs - atom type
        '''
        num_atoms = dist_graph.shape[0]
        atom_emb = self.atom_emb(atom_emb)
        # for atom type prediction
        atom_emb_type = self.atom_type_gnn(dist_graph, dist_exp, atom_emb)
        atom_type_pred = self.atom_classifier(atom_emb_type)

        # for atom position prediction
        atom_emb_pos = self.atom_pos_gnn(dist_graph, dist_exp, atom_emb)
        gaussians = th.rand((num_atoms, self.gaussian_size))
        atom_pos_vae = self.atom_pos_vae(gaussians, atom_emb_pos)
        atom_pos_pred = self.atom_pos_linear(atom_pos_vae)

        return atom_type_pred, atom_pos_pred, atom_pos_vae


class SSLMoleculeLoss(nn.Module):
    def __init__(self):
        super().__init__()