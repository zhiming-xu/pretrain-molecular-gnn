import scipy.io as si
import numpy as np


def load_qm9(datafile):
    mat = si.loadmat(datafile)
    # only use R and Z, and predict T
    Rs, Zs = mat['R'], mat['Z']
    new_Rs, new_Zs, Ds = [], [], []
    Ts = mat['T']
    
    for i, Z in enumerate(Zs):
        non_zeros = np.nonzero(Z)[0]
        new_Z = Zs[i][non_zeros]
        new_Zs.append(new_Z)
        new_R = Rs[i][non_zeros]
        new_Rs.append(new_R)
        abs_D = new_R[None, :, :].repeat(new_R.shape[0], axis=0) - \
                new_R[:, None, :].repeat(new_R.shape[0], axis=1)
        D = np.linalg.norm(abs_D, axis=-1)
        Ds.append(D)

    return new_Rs, Zs.tolist(), Ds

