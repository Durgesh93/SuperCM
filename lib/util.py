import umap
import wandb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression 
import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate
import time
from lib import rn,wrn
import argparse


class EMA:
    """
    Implementation from https://fyubang.com/2019/06/01/ema/
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def load(self, ema_model):
        for name, param in ema_model.named_parameters():
            self.shadow[name] = param.data.clone()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}



def create_model(name,num_classes,transform_fn,device,T=1,set_cenmode='super'):
    if name == 'r18':
        return rn.ResNet18_CM(num_classes,transform_fn,T,set_cenmode).to(device)
    elif name == 'wr28-2':
        return wrn.WRN_CM(2,num_classes, transform_fn,T,set_cenmode).to(device)
    elif name == 'wr28-8':
        return wrn.WRN_CM(8,num_classes, transform_fn,T,set_cenmode).to(device)
    
def gen_color_list(num_c,cmap_name='Spectral'):
    cmap=get_cmap(cmap_name)
    rgba= cmap(np.linspace(0,1,num=num_c))
    hex_list = [to_hex(c)for c in rgba]
    return hex_list


def evaluate(loader,model,ema,device):
    
    logits_dist = []
    preds       = []
    ema_preds   = []
    lbls        = []
    
    acc       = 0
    ema_acc   = 0

    with torch.no_grad():
        model.eval()
        for j, data in enumerate(loader):
            input, target = data
            input, target = input.to(device).float(), target.to(device).long()

            output = model(input)
            ema.apply_shadow()
            output_ema = model(input)
            ema.restore()
            logits_dist.append(output[1])
            pred_label = output[1].max(1)[1]
            pred_label_ema = output_ema[1].max(1)[1]
            preds.append(pred_label.cpu())
            ema_preds.append(pred_label_ema.cpu())
            lbls.append(target.cpu())

        preds       = torch.cat(preds,dim=0)
        ema_preds   = torch.cat(ema_preds,dim=0)
        lbls        = torch.cat(lbls,dim=0)
        logits_dist = torch.cat(logits_dist,dim=0)

        acc = (preds == lbls).float().mean()
        inc_idx = (preds != lbls)
        ema_acc = (ema_preds == lbls).float().mean()
        cert    = cer(logits_dist.cpu(),inc_idx.cpu())
    model.train() 
    return acc.item(),ema_acc.item(),cert.item()


def set_centroids(i,model,imgs,lbls):
    with torch.no_grad():
        model.eval()
        _, super_z= model(x=imgs, return_feature=True)
        cm_module = model.cm_module
        all_k   =  torch.unique(lbls)
        for k in all_k:
            m_z = super_z[ lbls == k ].detach().mean(0)
            cm_module.mu.weight.data[:,k] = (m_z +(i-1)*cm_module.mu.weight.data[:,k])/i
    model.train()
   

def filter_high_conf_psed(x,model,th):
    with torch.no_grad():
        model.eval()    
        out        = model(x)[2]
        logit,pred = out.max(-1)
        idx        = (logit > th)
        x          = x[idx]
        pred       = pred[idx]
    model.train()
    return x,pred

    
def acc(y_true,y_pred):
    return accuracy_score(y_true,y_pred)

def cer(ypred_logits,idx):
    ypred_logits = ypred_logits[idx]
    uniform = torch.full(ypred_logits.shape,1/(ypred_logits.shape[1]))
    ysmax = torch.log_softmax(ypred_logits,dim=-1)
    return -torch.mean(uniform*ysmax)

def H(p):
    return torch.sum(-p*torch.log2(p),axis=-1)


def random_stratified_split(labels,frac):

    if frac <=1:
        sup_indx=pd.DataFrame({'data_index':np.arange(0,len(labels)),
                                'targets':labels}
                            ).groupby('targets').sample(frac=frac)['data_index'].to_numpy()
    else:
        sup_indx=pd.DataFrame({'data_index':np.arange(0,len(labels)),
                                'targets':labels}
                            ).groupby('targets').sample(n=frac)['data_index'].to_numpy()

    unsup_indx= np.setdiff1d(np.arange(0,len(labels)),sup_indx)

    return list(sup_indx),list(unsup_indx)


def get_UMAP_fig(loader,model,device,colorlist):

    feats=[]
    preds=[]
    lbls =[]

    with torch.no_grad():
        model.eval()
        for j, data in enumerate(loader):
            input, target = data
            input, target = input.to(device).float(), target.to(device).long()
            output,feat = model(input,return_feature=True)
            pred_label = output[1].max(1)[1]
            preds.append(pred_label.cpu())
            lbls.append(target.cpu())
            feats.append(feat.cpu())

        mu      = model.cm_module._mu().cpu()


    feats   = torch.cat(feats,dim=0).numpy()
    y_pred  = torch.cat(preds,dim=0).numpy()
    y_true  = torch.cat(lbls,dim=0).numpy()
    mu      = mu.numpy()

    # idx,_ = random_stratified_split(y_true,frac=0.5)
    # feats = feats[idx]
    # y_pred = y_pred[idx]
    # y_true = y_true[idx]
   

    reducer = umap.UMAP()
    reducer.fit(feats)
    num_classes = np.unique(y_true)
    true_cen_2d = []
    
    for c in num_classes:
        cfeats = reducer.transform(feats[y_true == c])
        true_cen_2d.append(np.mean(cfeats,axis=0))


    clus_cen_2d = []
    num_clusters = np.unique(y_pred)
    

    cluswise_feats_2d={}

    for c in num_clusters:
        clusfeats = reducer.transform(feats[y_pred == c])
        cluswise_feats_2d[c]=clusfeats
        clus_cen_2d.append(np.mean(clusfeats,axis=0))
    
    mu_2d             = reducer.transform(mu)
    true_cen_2d       = np.stack(true_cen_2d,axis=0)
    clus_cen_2d       = np.stack(clus_cen_2d,axis=0)
   
    
    fig = plt.figure()
    
    for c in num_clusters:
        plt.scatter(cluswise_feats_2d[c][:,0],cluswise_feats_2d[c][:,1],marker='o',c=colorlist[c])

    cluswise_feats_2d_copy ={str(k):v for k,v in cluswise_feats_2d.items()}
    np.savez('./dirs/files/umap_{}.npz'.format(wandb.run.id),**cluswise_feats_2d_copy)

    # plt.scatter(true_cen_2d[:,0],true_cen_2d[:,1],marker='+',facecolors='none',label='true',c='#000000')
    # plt.scatter(clus_cen_2d[:,0],clus_cen_2d[:,1],marker='+',label='pred',c='#000000')
    plt.scatter(mu_2d[:,0],mu_2d[:,1],marker='^',label='mu',c='#000000')
    
    model.train()
    
    return fig


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def print_params_dict(d,logger):
    logger.log_str(tabulate([[k,v] for k,v in d.items()],headers=['Parameter','Value']))



def remain_hrs(p,st):
    if p == 0:
        return 100
    else:
        et      = time.time()
        sec     = et-st
        rate    = p/sec
        rem_sec = (1-p)/rate
        rem_hrs = rem_sec/3600
        return rem_hrs


