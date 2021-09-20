import scipy.io as si
import numpy as np


def load_qm9(datafile):
    mat = si.loadmat(datafile)
    # only use R and Z, and predict T
    Rs, Zs = mat['R'].tolist(), mat['Z']
    Ds = [] # pairwise distance
    Ts = mat['T']
    
    for i, Z in enumerate(Zs):
        non_zeros = np.nonzero(Zs)[0]
        Rs[i] = Rs[i][non_zeros]
        abs_D = Rs[i][None, :, :].repeat(5, axis=0)-Rs[i][:, None, :].repeat(5, axis=1)
        D = np.linalg.norm(abs_D, axis=-1)
        Ds.append(D)

    return Rs, Zs, Ds

        
