import logging
import wandb
import os
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import shutil
import uuid

class Logger:

    def __init__(self,conf,mode='w',run_name='',dir=''):
        
        self.logmode    = mode
        self.dir        = dir
        self.USE_WANDB  = False
        self.USE_TB     = False
        self.USE_FILE   = False
       
        self.metric_dict = {}

        if 'w' in self.logmode:
            self.USE_WANDB = True
        if 't' in self.logmode:
            self.USE_TB = True
        if 'f' in self.logmode:
            self.USE_FILE = True
        
    
        uni_str= str(uuid.uuid4())[:4]
        self.run_name       =  run_name
        self.log_dirname    = "{}-{}-{}".format(self.dir,self.run_name,uni_str)
        

        if self.USE_WANDB:
            wandb.init(config=conf,entity=os.environ['WB_ENTITY_NAME'],project=os.environ['WB_PROJECT_NAME'],dir='./dirs')
            wandb.run.name = wandb.run.name.replace('-'+self.run_name,'')
            wandb.run.name = wandb.run.name+'-'+self.run_name
            
            
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if self.USE_FILE:
            self.log_dirnamef = os.path.join('./dirs/files',self.log_dirname)
            os.makedirs(self.log_dirnamef,exist_ok=True)
            fh = logging.FileHandler(os.path.join(self.log_dirnamef,'log.txt'))
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        if self.USE_TB:
            self.log_dirnamet = os.path.join('./dirs/tb',self.log_dirname)
            os.makedirs(self.log_dirnamet,exist_ok=True)
            self.tbwriter = SummaryWriter(log_dir=self.log_dirnamet)

        self.logger = logger
    
    

    def log_scalars(self,data_dict,it):
        
        for k,v in data_dict.items():
            if k not in self.metric_dict:
                self.metric_dict[k]=[]
            self.metric_dict[k].append(v)

        if self.USE_WANDB:
            wandb.log(data=data_dict,step=it)

        if self.USE_TB:
            for k,v in data_dict.items():
                self.tbwriter.add_scalar(k,v,global_step=it)


    def print(self,it):
        log_str="iter:{:06d} | ".format(it)
        for k,v in self.metric_dict.items():
            k = k.split('/')[1]
            log_str+='{}={:.2f}, '.format(k,v[-1])
        log_str = log_str[:-2]
        self.logger.info(log_str)

    def log_fig(self,fig_dict,it):
       
        if self.USE_WANDB:
            wandb.log(data=fig_dict,step=it)

        if self.USE_TB:
            for fname,fig in fig_dict:
                self.tbwriter.add_figure(fname,fig,global_step=it)

        if self.USE_FILE:
            for fname,fig in fig_dict:
                fig.save_fig(os.path.join(self.log_dirnamef,fname.replace('/','-')+'-iter{}'.format(it)+'.jpg'))

    def log_str(self,s):
        self.logger.info(s)