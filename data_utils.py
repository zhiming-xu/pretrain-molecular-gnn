import scipy.io as si
import numpy as np


z2id = {1:0, 6: 1, 7:2, 8:3, 16: 4}


def load_qm7(datafile):
    mat = si.loadmat(datafile)
    # only use R and Z, and predict T
    Rs, Zs = mat['R'], mat['Z']
    new_Rs, new_Zs, Ds = [], [], []
    Ts = mat['T']
    
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

    return new_Rs, new_Zs, Ds

