import numpy as np
import torch
import torch.nn as nn
np.random.seed(131)
torch.manual_seed(131)

# Please see https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm for the theory of baum welch
# forward pass of baum welch 
def dis_forward_pass(prob_all_s,prob_initial,prob_all_h,prob_all_y):
    T=prob_all_s.shape[1]
    N=prob_all_s.shape[2]
    forward_prob=torch.ones((prob_all_s[:,:,:,0].shape),device=prob_all_s.device)
    for j in range(T):
        if j==0:
            forward_prob_each=prob_initial+prob_all_h[:,j]+prob_all_y[:,j,None]
            forward_prob[:,j]=forward_prob_each-torch.logsumexp(forward_prob_each,1)[:,None]
        else:
            np_=prob_all_s[:,j,:]+forward_prob[:,j-1][:,None,:]
            forward_prob_each=torch.logsumexp(prob_all_h[:,j,:,None]+prob_all_y[:,j,None,None]+np_,axis=-1)

            forward_prob[:,j]=forward_prob_each-torch.logsumexp(forward_prob_each,1)[:,None]
    return forward_prob
# backward pass of baum welch             
def dis_backward_pass(prob_all_s,prob_initial,prob_all_h,prob_all_y):
    T=prob_all_s.shape[1]
    N=prob_all_s.shape[2]
    backward_prob=torch.ones((prob_all_s[:,:,:,0].shape),device=prob_all_s.device)
    for j in range(T-1,-1,-1):
        if j==T-1:
            backward_prob_each=torch.zeros((prob_all_s.shape[0],N),device=prob_all_s.device)
            backward_prob[:,j]=backward_prob_each
        else:
            np_=backward_prob[:,j+1,:,None]+prob_all_s[:,j+1,:,:]+prob_all_h[:,j+1,:,None]+prob_all_y[:,j+1,None,None]
            backward_prob_each=torch.logsumexp(np_,axis=1)
            backward_prob[:,j]=backward_prob_each-torch.logsumexp(backward_prob_each,axis=1)[:,None]
    return backward_prob

def get_gamma(forward_prob,backward_prob,prob_all_s,prob_all_h,prob_all_y):

    gamma1 = forward_prob[:,0,:] + backward_prob[:,0,:]  # (B, N)
    gamma1 = gamma1 - torch.logsumexp(gamma1, 1)[:, None]


    delta1 = (forward_prob[:, :-1, :, None]    # (B, T-1, N, 1)
            + prob_all_s[:, :-1]               # (B, T-1, N, N)
            + backward_prob[:, 1:, None, :]    # (B, T-1, 1, N)
            + prob_all_h[:, 1:, None, :]       # (B, T-1, 1, N)
            + prob_all_y[:, 1:, None, None])   # (B, T-1, 1, 1)


    B, Tm1, N, _ = delta1.shape
    sum_delta1 = torch.logsumexp(delta1.reshape(B, Tm1, N * N), dim=2)  # (B, T-1)
    delta1 = delta1 - sum_delta1[:, :, None, None]

    return gamma1, delta1