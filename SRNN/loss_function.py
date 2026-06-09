import numpy as np
import torch
import torch.nn as nn
import math
np.random.seed(131)
torch.manual_seed(131)

def get_loss(gamma1,delta1,prob_ini,prob_all_s,prob_all_h,prob_all_y):
    # see Equation 15 of our paper
    t1=torch.sum(torch.exp(gamma1)*(prob_ini+prob_all_h[:,0]+prob_all_y[:,0][:,None]))

    # Vectorized: replaces for loop over time steps
    # delta1 shape: (B, T-1, N, N)
    # prob_all_s[:, 1:] shape: (B, T-1, N, N)
    # prob_all_h[:, 1:] shape: (B, T-1, N) -> need (B, T-1, 1, N) for broadcasting
    # prob_all_y[:, 1:] shape: (B, T-1) -> need (B, T-1, 1, 1) for broadcasting
    terms = (prob_all_s[:, 1:]
           + prob_all_h[:, 1:, None, :]
           + prob_all_y[:, 1:, None, None]
           )
    t2 = torch.sum(torch.exp(delta1) * terms)

    return t1,t2

def get_cross_entropy(pos,pri):
    # Fully vectorized: replaces double nested loop over (k, i)
    return (pos * pri).sum()


