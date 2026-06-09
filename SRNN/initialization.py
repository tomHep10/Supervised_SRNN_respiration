import numpy as np

np.random.seed(131)


def one_hot(hmm_z_all, num_tv):
    pp = np.zeros((hmm_z_all.shape[0], hmm_z_all.shape[1], num_tv))
    for kk in range(hmm_z_all.shape[0]):
        t_all = hmm_z_all[kk]
        for i in range(hmm_z_all.shape[1]):
            pp[kk, i, int(t_all[i])] = 1
    return pp
