import scipy.io as si
import numpy as np
import torch as th
from torch.utils.data import Dataset
from torch_geometric.transforms import BaseTransform


z2id = {1:0, 6: 1, 7:2, 8:3, 16: 4}


def load_qm7_dataset(datafile):
    mat = si.loadmat(datafile)
    # only use R and Z, and predict T
    Rs, Zs, Ts = mat['R'], mat['Z'], mat['T'].swapaxes(1, 0)
    # randomly shuffle indices
    shuffle_idx = np.arange(Rs.shape[0])
    np.random.shuffle(shuffle_idx)
    Rs, Zs, Ts = Rs[shuffle_idx], Zs[shuffle_idx], Ts[shuffle_idx] 
    new_Rs, new_Zs, Ds = [], [], []
    Ts = (Ts - Ts.mean()) / Ts.std()
    
    for i, Z in enumerate(Zs):
        non_zeros = np.nonzero(Z)[0]
        new_Z = Zs[i][non_zeros]
        new_Z = np.array(list(map(z2id.__getitem__, new_Z)), dtype=int)
        new_Zs.append(new_Z)
        new_R = Rs[i][non_zeros]
        new_Rs.append(new_R)
        abs_D = new_R[None, :, :].repeat(new_R.shape[0], axis=0) - \
                new_R[:, None, :].repeat(new_R.shape[0], axis=1)
        D = np.linalg.norm(abs_D, axis=-1)
        Ds.append(D)

    return new_Rs, new_Zs, Ds, Ts


class QM7Dataset(Dataset):
    def __repr__(self):
         return "DataContainer"
    def __init__(self, filename):
        #read in data
        extension_name = filename.split('.')[-1]
        if extension_name == 'npz':
            dictionary = np.load(filename)
        elif extension_name == 'mat':
            dictionary = si.loadmat(filename) 
        #number of atoms
        if 'N' in dictionary: 
            self._N = dictionary['N'].flatten()
        else:
            self._N = None
        #atomic numbers/nuclear charges
        if 'Z' in dictionary:
            self._Z = dictionary['Z'] 
        else:
            self._Z = None
        #reference dipole moment vector
        if 'D' in dictionary: 
            self._D = dictionary['D'] 
        else:
            self._D = None
        #reference total charge
        if 'Q' in dictionary: 
            self._Q = dictionary['Q'] 
        else:
            self._Q = None
        #reference atomic charges
        if 'Qa' in dictionary: 
            self._Qa = dictionary['Qa'] 
        else:
            self._Qa = None
        #positions (cartesian coordinates)
        if 'R' in dictionary:     
            self._R = dictionary['R'] 
        else:
            self._R = None
        #reference energy
        if 'E' in dictionary:
            self._E = dictionary['E'].flatten()
        else:
            self._E = None
        #reference atomic energies
        if 'Ea' in dictionary:
            self._Ea = dictionary['Ea']
        else:
            self._Ea = None
        #reference forces
        if 'F' in dictionary:
            self._F = dictionary['F'] 
        else:
            self._F = None

        #maximum number of atoms per molecule
        self._N_max    = self.Z.shape[1] 

        #construct indices used to extract position vectors to calculate relative positions 
        #(basically, constructs indices for calculating all possible interactions (excluding self interactions), 
        #this is a naive (but simple) O(N^2) approach, could be replaced by something more sophisticated) 
        self._idx_i = np.empty([self.N_max, self.N_max-1],dtype=int)
        for i in range(self.idx_i.shape[0]):
            for j in range(self.idx_i.shape[1]):
                self._idx_i[i,j] = i

        self._idx_j = np.empty([self.N_max, self.N_max-1],dtype=int)
        for i in range(self.idx_j.shape[0]):
            c = 0
            for j in range(self.idx_j.shape[0]):
                if j != i:
                    self._idx_j[i,c] = j
                    c += 1

    @property
    def N_max(self):
        return self._N_max

    @property
    def N(self):
        return self._N

    @property
    def Z(self):
        return self._Z

    @property
    def Q(self):
        return self._Q

    @property
    def Qa(self):
        return self._Qa

    @property
    def D(self):
        return self._D

    @property
    def R(self):
        return self._R

    @property
    def E(self):
        return self._E

    @property
    def Ea(self):
        return self._Ea
    
    @property
    def F(self):
        return self._F

    #indices for atoms i (when calculating interactions)
    @property
    def idx_i(self):
        return self._idx_i

    #indices for atoms j (when calculating interactions)
    @property
    def idx_j(self):
        return self._idx_j

    def __len__(self): 
        return self.Z.shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, np.int64):
            idx = [idx]

        data = {'E':         [],
                'Ea':        [],    
                'F':         [],
                'Z':         [],
                'D':         [],
                'Q':         [],
                'Qa':        [],
                'R':         [],
                'idx_i':     [],
                'idx_j':     [],
                'batch_seg': [],
                'offsets'  : []
                }

        Ntot = 0 #total number of atoms
        Itot = 0 #total number of interactions
        for k, i in enumerate(idx):
            N = self.N[i] #number of atoms
            I = N*(N-1)   #number of interactions
            #append data
            if self.E is not None:
                data['E'].append(self.E[i])
            else:
                data['E'].append(np.nan)
            if self.Ea is not None:
                data['Ea'].extend(self.Ea[i,:N].tolist())
            else:
                data['Ea'].extend([np.nan])
            if self.Q is not None:
                data['Q'].append(self.Q[i])
            else:
                data['Q'].append(np.nan)
            if self.Qa is not None:
                data['Qa'].extend(self.Qa[i,:N].tolist())
            else:
                data['Qa'].extend([np.nan])
            if self.Z is not None:
                data['Z'].extend(self.Z[i,:N].tolist())
            else:
                data['Z'].append(0)
            if self.D is not None:
                data['D'].extend(self.D[i:i+1,:].tolist())
            else:
                data['D'].extend([[np.nan,np.nan,np.nan]])
            if self.R is not None:
                data['R'].extend(self.R[i,:N,:].tolist())
            else:
                data['R'].extend([[np.nan,np.nan,np.nan]])
            if self.F is not None:
                data['F'].extend(self.F[i,:N,:].tolist())
            else:
                data['F'].extend([[np.nan,np.nan,np.nan]])
            data['idx_i'].extend(np.reshape(self.idx_i[:N,:N-1]+Ntot,[-1]).tolist())
            data['idx_j'].extend(np.reshape(self.idx_j[:N,:N-1]+Ntot,[-1]).tolist())
            #offsets could be added in case they are need
            data['batch_seg'].extend([k] * N)
            #increment totals
            Ntot += N
            Itot += I

        for k, v in data.items():
            if k in ['Z', 'idx_i', 'idx_j']:
                data[k] = th.LongTensor(v)
            else:
                data[k] = th.FloatTensor(v)

        return data


class DistanceAndPlanarAngle(BaseTransform):
    def __init__(self) -> None:
       super().__init__()
        
    def __call__(self, data):
        # calculate bond angle
        (src, dst), pos = data.edge_index, data.pos
        adj_atoms = {}
        for s, d in zip(src, dst):
            if s.item() not in adj_atoms:
                adj_atoms[s.item()] = [d.item()]
            else:
                adj_atoms[s.item()].append(d.item())
        ijk = []
        for u, vs in adj_atoms.items():
            if len(vs) > 1:
                for i in range(len(vs)):
                    for j in range(i+1, len(vs)):
                        ijk.append([vs[i], u, vs[j]])
        ijk = th.LongTensor(ijk)
        loc = pos[ijk]
        vij = loc[:, 1] - loc[:, 0]
        vik = loc[:, 1] - loc[:, 2]
        bond_cos = (vij*vik).sum(dim=-1, keepdims=True)/th.norm(vij, dim=-1, keepdim=True) \
                                                       /th.norm(vik, dim=-1, keepdim=True)
        bond_cos = th.where(bond_cos<-1, -th.ones_like(bond_cos), bond_cos)
        bond_cos = th.where(bond_cos>1, th.ones_like(bond_cos), bond_cos)
        bond_angle = th.acos(bond_cos)
        data.idx_ijk, data.bond_angle = ijk, bond_angle
        # calculate bond length
        dist = th.norm(pos[src] - pos[dst], p=2, dim=-1).view(-1, 1)
        data.bond_length = dist
        return data