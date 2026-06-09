import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from SRNN import loss_function
from SRNN import initialization
from sklearn.metrics import mean_squared_error
from sklearn.metrics import balanced_accuracy_score
import time
from SRNN import utils

np.random.seed(131)
torch.manual_seed(131)



def train_(model,rnninfer,optimizer,optimizer_rnn,scheduler,scheduler_rnn,X_train,y_train,label_train,X_test,y_test,label_test,num_tv,coef_cross,epochs,batch_size,save_name,save_folder,save_fold,device):
    mse_all=np.ones(epochs)
    error_all=np.ones(epochs)
    mse_all_test=np.ones(epochs)
    error_all_test=np.ones(epochs)
    loss_save=np.ones(epochs)
    start_time=time.time()
    pos_test_save_all=np.zeros((epochs,y_test.shape[0],y_test.shape[1],num_tv))

    # cs_criterion = cs_criterion.to(device)
    pp=initialization.one_hot(label_train.cpu().detach().numpy()[:,:,0],num_tv)
    pp=torch.tensor(pp,device=device)

    tds = TensorDataset(X_train,y_train,pp)
    data_loader = DataLoader(tds, batch_size=batch_size, shuffle=True, drop_last=False)

    loss_neural_save=np.ones(epochs)
    loss_private_save=np.ones(epochs)

    for epoch in range(epochs):
        model.train()
        rnninfer.train()

        loss_print=0
        loss_neural_print=0
        loss_private_print=0
        for X_train_batch, y_train_batch, pp_batch in data_loader:
            optimizer.zero_grad()
            optimizer_rnn.zero_grad()
 
            infer_dist,inferred_h,mean_out=rnninfer(y_train_batch)
           
    
            global_latent=inferred_h
            
            prob_ini,prob_all_s,prob_all_h,prob_all_y,gamma1,delta1,fwp,bwp = model(X_train_batch,y_train_batch,global_latent,global_latent,device)
            t1,t2=loss_function.get_loss(gamma1,delta1,prob_ini,prob_all_s,prob_all_h,prob_all_y)
            posterior_lk=model.get_posterior_lk(fwp,bwp)
            
            
            cross_en=loss_function.get_cross_entropy(posterior_lk,pp_batch)

            loss_all=-(t1.mean()+t2.mean()+coef_cross*(cross_en)+infer_dist.entropy().mean())
            loss_all.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(rnninfer.parameters(), 1.0)

            
            optimizer.step()
            optimizer_rnn.step()

            loss_print+=loss_all

            
            
        if epoch%100==0:
            print(f"Epoch {epoch+1}/{epochs}, loss = {loss_all}")
            end_time = time.time()
            if epoch!=0:
                utils.compute_time(start_time,end_time,epochs,epoch+1)
        loss_save[epoch]=loss_print.item()



        scheduler.step()
        scheduler_rnn.step()

        y_pred_train,_,pos_train,_,_,_,_,_=eval_(model,rnninfer,X_train,y_train,device)

        y_pred_test,_,pos_test,_,_,_,_,_=eval_(model,rnninfer,X_test,y_test,device)
        pos_test_save_all[epoch]=pos_test



        torch.save({
            'num_tv':num_tv,


            'coef_cross':coef_cross,

            'y_test':y_test.cpu().detach().numpy(),
            'X_test':X_test.cpu().detach().numpy(),
            'label_test':label_test.cpu().detach().numpy(),

            'model_state_dict': model.state_dict(),
            'rnninfer_state_dict': rnninfer.state_dict(),

            'optimizer_state_dict': optimizer.state_dict(),
            'mse_all':mse_all,
            'error_all':error_all,
            'mse_all_test':mse_all_test,
            'error_all_test':error_all_test,
            'loss_train':loss_save,
            'loss_neural_train':loss_neural_save,
            'loss_private_train':loss_private_save,
            'pos_test_all':pos_test_save_all,
            }, save_folder+'/autosave_'+save_name+'_model_hidden'+str(model.hidden_shape)+'_fold_'+str(save_fold)+'.pt')
        
    return model,rnninfer,mse_all,error_all,mse_all_test,error_all_test,loss_save,pos_test_save_all


def eval_(model,rnninfer,X_test,y_test,device):
    model.eval()
    rnninfer.eval()

    infer_dist_test,inferred_h_test,mean_out_test=rnninfer(y_test)
   

    global_latent_test=inferred_h_test
    
    prob_ini_test,prob_all_s_test,prob_all_h_test,prob_all_y_test,gamma1_test,delta1_test,fwp_test,bwp_test = model(X_test,y_test,global_latent_test,global_latent_test,device)
    t1_test,t2_test=loss_function.get_loss(gamma1_test,delta1_test,prob_ini_test,prob_all_s_test,prob_all_h_test,prob_all_y_test)
    posterior_lk_test=model.get_posterior_lk(fwp_test,bwp_test)
    pos_test=posterior_lk_test.cpu().detach().numpy()

    emission_mean=model.emission(global_latent_test)
    covariance_matrix=(1e-20)*torch.eye(model.input_shape,device=device)
    emission_dist=torch.distributions.multivariate_normal.MultivariateNormal(emission_mean[:,:,:], covariance_matrix)
    y_pred_test=emission_dist.sample().cpu().detach().numpy()

    
    return y_pred_test,y_pred_test,pos_test,global_latent_test,global_latent_test,global_latent_test,global_latent_test,global_latent_test

def compute_metric(y_train,y_pred_train,pos_train):
    mse_=mean_squared_error(y_pred_train.reshape(-1,y_pred_train.shape[2]),y_train.cpu().detach().numpy().reshape(-1,y_train.shape[2]))
    error_=np.ones(2)
    return mse_,error_

def compute_error(pos_,truth_):
    srnn_acc_curve=np.ones(pos_.shape[0])
    for i in range(pos_.shape[0]):
        srnn_acc_curve[i]=balanced_accuracy_score(truth_[i], pos_[i])
    return 1-srnn_acc_curve
