import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import math
from SRNN import baum_welch

# comment this if not necessary to reproduce
np.random.seed(131)
torch.manual_seed(131)
############################################

# Precomputed constant for MVN log-prob with diagonal covariance
_LOG_2PI = math.log(2 * math.pi)

def _diag_mvn_log_prob(x, mean, var):
    """
    Fast log-prob of multivariate normal with covariance = var * I.
    Equivalent to MultivariateNormal(mean, var*I).log_prob(x) but much faster.
    x, mean: (..., D)
    var: scalar
    Returns: (...)
    """
    d = x.shape[-1]
    diff = x - mean
    return -0.5 * (d * _LOG_2PI + d * math.log(var) + (diff * diff).sum(-1) / var)


### Emission Model ###
class Emission(nn.Module):
    def __init__(self,input_shape,hidden_shape):
        super(Emission, self).__init__()
        self.input_shape=input_shape
        self.hidden_shape=hidden_shape
        self.fc1 = nn.Linear(self.hidden_shape,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64,self.input_shape)
        self.relu=nn.ReLU()

    def forward(self, sampled_h_):
        emission_mean=self.fc1(sampled_h_)
        emission_mean=self.relu(emission_mean)
        emission_mean=self.fc2(emission_mean)
        emission_mean=self.relu(emission_mean)
        emission_mean=self.fc3(emission_mean)
        return emission_mean



### SRNN Model ###
class Model(nn.Module):
    def __init__(self,input_shape,num_tv,hidden_shape,neural_private_shape):
        super(Model, self).__init__()
        self.input_shape=input_shape
        self.hidden_shape=hidden_shape
        self.neural_private_shape=neural_private_shape
        self.num_tv=num_tv
        self.rnns = nn.ModuleList([nn.RNN(input_size=self.input_shape, hidden_size=self.hidden_shape, num_layers=1, batch_first=True) for i in range(self.num_tv)]) # SRNNs
        self.emission = Emission(self.input_shape,self.hidden_shape) # Emission networks
        self.transitions = nn.RNNCell(self.hidden_shape, self.num_tv*self.num_tv) # Transition networks
        self.initials = nn.Parameter(torch.randn(self.num_tv), requires_grad=True) # Initial states
        

    def forward(self, x_input,y_train,sampled_h,neural_final,device):

        batches = int(x_input.shape[0])
        T = x_input.shape[1]
        dtype = torch.float32
        h0_transitions=torch.zeros([batches,self.num_tv*self.num_tv],device=device)

        # --- Vectorized prob_initial: no batch loop needed ---
        log_initial = self.initials - torch.logsumexp(self.initials, 0)
        prob_initial = log_initial.unsqueeze(0).expand(batches, -1)  # (B, num_tv)

        # Pre-allocate on device to avoid CPU-GPU transfers
        prob_all_s=torch.zeros((batches, T, self.num_tv, self.num_tv), device=device)
        prob_all_h=torch.zeros((batches, T, self.num_tv), device=device)
        prob_all_y=torch.zeros((batches, T), device=device)


        # Pre-compute identity matrix once (used at j=0)
        eye_matrix = torch.eye(self.num_tv, device=device)  # (num_tv, num_tv)

        # Variance for the generative model
        h_var = 1e-4
        y_var = 1e-4

        for j in range(T): # Temporal loop

            if j==0: # First time step
                # Vectorized: all states get same log-prob at t=0 (identity covariance)
                # P(h_0|h_0) with var=1e-4 => log_prob for each state i
                for i in range(self.num_tv):
                    prob_all_h[:,j,i] = _diag_mvn_log_prob(sampled_h[:,j,:], sampled_h[:,0,:], h_var)
                # Vectorized: set identity for all batches at once
                prob_all_s[:,j] = eye_matrix.unsqueeze(0)  # broadcasts to (B, num_tv, num_tv)

            else:
                trans_prob=torch.reshape(self.transitions(sampled_h[:,j-1,:],h0_transitions),(-1,self.num_tv,self.num_tv))
                prob_all_s[:,j]=trans_prob-torch.logsumexp(trans_prob,axis=1)[:,None,:]

                # Run all num_tv RNNs and compute log-probs
                h_prev = sampled_h[:,j-1,:].unsqueeze(0).contiguous()  # (1, B, H) - compute once
                x_slice = x_input[:,j:j+1,:]  # (B, 1, input_shape) - compute once
                h_target = sampled_h[:,j,:]  # (B, H) - compute once

                for i in range(self.num_tv):
                    x_out, _ = self.rnns[i](x_slice, h_prev)
                    prob_all_h[:,j,i] = _diag_mvn_log_prob(h_target, x_out[:,0,:], h_var)

            # Emission log-probs: use fast diagonal MVN instead of constructing distribution
            emission_mean=self.emission(neural_final[:,j:j+1,:])
            prob_all_y[:,j] = _diag_mvn_log_prob(y_train[:,j,:], emission_mean[:,0,:], y_var)

            
        # print(prob_all_s.device)
        # print(prob_initial.device)
        # print(prob_all_h.device)
        # print(prob_all_y.device)
        forward_prob=baum_welch.dis_forward_pass(prob_all_s,prob_initial,prob_all_h,prob_all_y)
        backward_prob=baum_welch.dis_backward_pass(prob_all_s,prob_initial,prob_all_h,prob_all_y)
        gamma1,delta1=baum_welch.get_gamma(forward_prob,backward_prob,prob_all_s,prob_all_h,prob_all_y)

        return prob_initial,prob_all_s,prob_all_h,prob_all_y,gamma1,delta1,forward_prob,backward_prob

    def get_posterior_lk(self,forward_prob,backward_prob):
        # Compute posterior log likelihood
        posterior_lk=forward_prob+backward_prob
        posterior_lk=posterior_lk-torch.logsumexp(posterior_lk,axis=2)[:,:,None]
        return posterior_lk
