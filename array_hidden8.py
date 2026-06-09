# -*- coding: utf-8 -*-
"""demo_lorenz


# Switching Recurrent Neural Networks (SRNNs) Tutorial on Lorenz Attractor

### Step 1: Package load
In this step, we load the packages of SRNNs. **Please note** the initializatins are very important to SRNNs as well as other SSM models to shorten training time and avoid stuck of training as we discussed in our paper. If you use HMM as a initialization, you have two options:
<br>
(1) We provide HMM package from Linderman SSM, you may **import ssm** to test whether the SSM package is installed. If so, you can specify the initialization method to be 'hmm' in step 5.
<br>
(2) If the ssm doesn't work for you or you would like to use your own initialization (e.g., your own HMM, your own labels, etc.), please change the initialization method to be 'defined'.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import scipy.io as sio
from SRNN import model_srnn
from SRNN import inference_network
from SRNN import train
import yaml
import argparse
from sklearn.model_selection import KFold

save_name='sim'  # overridden below if --mouse is passed
def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--num_tv", type=int, default=None)
    p.add_argument("--hidden_shape", type=int, default=None)
    p.add_argument("--mouse", type=str, default=None)
    return p.parse_args()

args = parse_args()
cfg = load_config(args.config)

if args.mouse is not None:
    cfg["experiment"]["data_path"] = f"./data/simulation.npy"
    cfg["experiment"]["data_label"] = f"./data/labels.npy"
    save_name = f"mh_{args.mouse}"

if args.fold is not None:
    cfg["experiment"]["fold"] = args.fold
if args.lr is not None:
    cfg["train"]["lr"] = args.lr
if args.epochs is not None:
    cfg["train"]["epochs"] = args.epochs
if args.num_tv is not None:
    cfg["model"]["num_tv"] = args.num_tv
if args.hidden_shape is not None:
    cfg["model"]["hidden_shape"] = args.hidden_shape


# Seeds
seed = int(cfg["experiment"].get("seed", 131))
np.random.seed(seed)
torch.manual_seed(seed)

# Device
device_cfg = cfg["system"].get("device", "auto")
if device_cfg == "auto":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device(device_cfg)

# dtype
dtype_str = cfg["system"].get("dtype", "float32")
dtype = torch.float32 if dtype_str == "float32" else torch.float64

save_folder=cfg["experiment"]["save_dir"]
save_fold=cfg["experiment"]["fold"]

"""### Step 2: Data load
In this step, we load and visualize the data of lorenz attractor.
"""

# sim_out = np.load(cfg["experiment"]["data_path"]) #You may change this 'y_c' to your own data, the data has to be in size of (#samples*#time points*#features).



y_c = np.load(cfg["experiment"]["data_path"],allow_pickle=True) #You may change this 'y_c' to your own data, the data has to be in size of (#samples*#time points*#features).
label_c=np.load(cfg["experiment"]["data_label"],allow_pickle=True)

"""We then split the data into training and testing, i.e., 17 trials in training and 1 trials in testing. 'jobid' is to set which trial in testing."""

jobid = save_fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kf.split(y_c))
train_idx, test_idx = splits[int(jobid)]

train_data = y_c[train_idx]
test_data = y_c[test_idx]
train_data_label = label_c[train_idx]
test_data_label = label_c[test_idx]


print(np.unique(train_data_label))
print(np.unique(test_data_label))
print("=" * 55)
print("Final output shapes:")
print(f"  neural_train   : {train_data.shape}   (trials, time, neurons)")
print(f"  neural_test    : {test_data.shape}    (trials, time, neurons)")
print(f"  labels_train   : {train_data_label.shape}  (trials, time, 1)")
print(f"  labels_test    : {test_data_label.shape}   (trials, time, 1)")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
X_train=torch.tensor(0*train_data,dtype=dtype,device=device)
y_train=torch.tensor(train_data,dtype=dtype,device=device)
y_test=torch.tensor(test_data,dtype=dtype,device=device)
X_test=torch.tensor(0*test_data,dtype=dtype,device=device)

label_train=torch.tensor(train_data_label,dtype=dtype,device=device)
label_test=torch.tensor(test_data_label,dtype=dtype,device=device)

"""### Step 3: Hyperparameters"""

input_shape=X_train.shape[2] # Input shape of SRNNs, but the models are input free.
num_tv = int(cfg["model"]["num_tv"])
hidden_shape = int(cfg["model"]["hidden_shape"])

bottleneck_shape=cfg["model"]["bottleneck_shape"]
neural_private_shape=cfg["model"]["neural_private_shape"]

epochs = int(cfg["train"]["epochs"])
lr = float(cfg["train"]["lr"])
coef_cross = float(cfg["train"]["coef_cross"])

batch_size = int(cfg["train"]["batch_size"])



"""### Step 4: Define SRNN and Inference Networks"""

model = model_srnn.Model(input_shape,num_tv,hidden_shape,neural_private_shape).to(device)
rnninfer=inference_network.RNNInfer(input_shape,hidden_shape).to(device)


# model = torch.compile(model)
# rnninfer=torch.compile(rnninfer)
# sharedecoder=torch.compile(sharedecoder)


"""### Step 5: Initialization"""

optimizer = torch.optim.Adam(
    list(model.parameters()) , lr=lr)

optimizer_rnn = torch.optim.Adam(
    list(rnninfer.parameters()), lr=lr)




# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.8)
scheduler_rnn = torch.optim.lr_scheduler.StepLR(optimizer_rnn, step_size=2000, gamma=0.8)




"""### Step 6: Training
"""
model_trained,rnninfer_trained,mse_all_train,error_all_train,mse_all_test,error_all_test,loss_all,pos_test_all=train.train_(model,
                                                                                                                    rnninfer,
                                                                                                                    
                                                                                                                    optimizer,
                                                                                                                    optimizer_rnn,
                                                                                                                    
                                                                                                                    scheduler,
                                                                                                                    scheduler_rnn,
                                                                                                                    
                                                                                                                    X_train,
                                                                                                                    y_train,
                                                                                                                    label_train,
                                                                                                                    X_test,
                                                                                                                    y_test,
                                                                                                                    label_test,
                                                                                                                    num_tv,
                                                                                                                    coef_cross,
                                                                                                                    epochs,
                                                                                                                    batch_size,
                                                                                                                    save_name,
                                                                                                                    save_folder,
                                                                                                                    save_fold,
                                                                                                                    device)

"""Now, we can test the SRNN after training.

### Step 7: Analysis
"""

# y_pred_test,pos_test,sampled_h_test=train.eval_(model_trained,rnninfer_trained,X_test,y_test,device)


torch.save({
            'num_tv':num_tv,
            'lr':lr,
            'bottleneck_shape':bottleneck_shape,
            'neural_private_shape':neural_private_shape,
            'coef_cross':coef_cross,
            'hidden_shape':hidden_shape,
            'y_train':y_train.cpu().detach().numpy(),
            'X_train':X_train.cpu().detach().numpy(),
            'label_train':label_train.cpu().detach().numpy(),
            'y_test':y_test.cpu().detach().numpy(),
            'X_test':X_test.cpu().detach().numpy(),
            'label_test':label_test.cpu().detach().numpy(),
            'model_state_dict': model_trained.state_dict(),
            'rnninfer_state_dict': rnninfer_trained.state_dict(),
            
            'optimizer_state_dict': optimizer.state_dict(),
            'mse_all':mse_all_train,
            'error_all':error_all_train,
            'mse_all_test':mse_all_test,
            'error_all_test':error_all_test,
            'loss_train':loss_all,
            'pos_test_all':pos_test_all,
            }, save_folder+'/'+save_name+'_model_hidden'+str(hidden_shape)+'_fold'+str(save_fold)+'.pt')




