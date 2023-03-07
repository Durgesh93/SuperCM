import os
seed = int(os.environ['SEED'])
import random
if seed == -1:
    seed = random.randint(0, 2**32)
random.seed(seed)
import numpy as np
np.random.seed(seed)
import torch
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False

import tempfile
dirname = os.path.join(tempfile._get_default_tempdir(),next(tempfile._get_candidate_names()))
os.makedirs(dirname,exist_ok=True)
os.environ['MIOPEN_USER_DB_PATH']=dirname


import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib import transform,util
from lib.config import config
import argparse,time,math
from lib.logger import Logger


parser = argparse.ArgumentParser()

parser.add_argument("--config",            required=True,                     type=str           )
parser.add_argument("--load_args",         required=True,                     type=util.str2bool )
parser.add_argument("--root",              default="./dirs/data_storage",     type=str           )
parser.add_argument("--validation",        default=25000,                     type=int           )
parser.add_argument("--plot_umap",         default=False,                     type=util.str2bool )
parser.add_argument("--run_name",          default='',                        type=str           )
parser.add_argument("--logmode",           default='w',                       type=str           )
parser.add_argument("--T",                 default=1,                         type=float         )
parser.add_argument("--set_cenmode",       default='super',                   type=str           )

parser.add_argument("--gamma",             default='-inf',                    type=float         )
parser.add_argument("--beta",              default='-inf',                    type=float         )
parser.add_argument("--lr",                default='-inf',                    type=float         )



args = parser.parse_args()

conf = config.get_config(args.config)

if args.load_args:
    conf = config.update_args(conf,args)

transform_fn    = transform.transform(*conf["transform"])

logger            = Logger(conf=conf,mode='w')
util.print_params_dict(conf,logger)

conf['beta']      = 10**float(conf['beta'])  if float(conf['beta']) > -5 else 0
conf['gamma']     = 10**float(conf['gamma']) if float(conf['gamma']) > -5 else 0
conf['lr']        = 10**float(conf['lr']) if float(conf['lr']) > -5 else 0


if torch.cuda.is_available():
    device = "cuda:"+str(os.environ.get('GPU','0'))
else:
    device = "cpu"


l_train_dataset = conf["dataset"](args.root, "l_train_{}".format(conf['nlabels']))
u_train_dataset = conf["dataset"](args.root, "u_train_{}".format(conf['nlabels']))
val_dataset     = conf["dataset"](args.root, "val_{}".format(conf['nlabels']))
test_dataset    = conf["dataset"](args.root, "test_{}".format(conf['nlabels']))

clist= util.gen_color_list(num_c=conf['num_classes'])
 
class RandomSampler(torch.utils.data.Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


if conf['beta']== 0 and conf['alg'] == 'supervised':
    l_loader = DataLoader(
        l_train_dataset, conf["batch_size"], drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), conf["iteration"] * conf["batch_size"])
    )
else:
    l_loader = DataLoader(
        l_train_dataset, conf["batch_size"]//2, drop_last=True,
        sampler=RandomSampler(len(l_train_dataset), conf["iteration"] * conf["batch_size"]//2)
    )


u_loader = DataLoader(
    u_train_dataset, conf["batch_size"]//2, drop_last=True,
    sampler=RandomSampler(len(u_train_dataset), conf["iteration"] * conf["batch_size"]//2)
)

val_loader = DataLoader(val_dataset, conf["batch_size"], shuffle=False, drop_last=False)
test_loader = DataLoader(test_dataset, conf["batch_size"], shuffle=False, drop_last=False)


model = util.create_model(conf['model'],conf["num_classes"],transform_fn,device,conf['T'],set_cenmode=conf['set_cenmode'])
ema   = util.EMA(model=model,decay=0.99)
ema.register()

optimizer = optim.Adam(model.parameters(), lr=conf['lr'])
trainable_paramters = sum([p.data.nelement() for p in model.parameters()])



if conf['alg'] == "VAT": 
    from lib.algs.vat import VAT
    ssl_obj = VAT(conf["eps"],conf["xi"], 1)

elif conf['alg'] == "PL": 
    from lib.algs.pseudo_label import PL
    ssl_obj = PL(conf["threashold"])
    

elif conf['alg'] == "MT": 
    from lib.algs.mean_teacher import MT
    t_model = util.create_model(conf['model'],conf["num_classes"],transform_fn,device,conf['T'],set_cenmode=conf['set_cenmode'])
    t_model.load_state_dict(model.state_dict())
    ssl_obj = MT(t_model, conf["ema_factor"])


elif conf['alg'] == "PI": 
    from lib.algs.pimodel import PiModel
    ssl_obj = PiModel()
    
elif conf['alg'] == "supervised":
    pass
else:
    raise ValueError("{} is unknown algorithm".format(conf['alg']))

from lib.algs.cm_loss import CM_loss
cm_obj = CM_loss(num_clusters=conf["num_classes"]).to(device)

iteration = 0
maximum_val_acc = 0
st = time.time()
logger.log_str('Global Seed set to               : {}'.format(seed))
logger.log_str("model                            : {}".format(conf['model']))
logger.log_str("trainable parameters             : {}".format(trainable_paramters))
logger.log_str("labeled data: {}, unlabeled data : {}, training data : {}".format(len(l_train_dataset), len(u_train_dataset), len(l_train_dataset)+len(u_train_dataset)))
logger.log_str("validation data: {}, test data   : {}".format(len(val_dataset), len(test_dataset)))



for l_data, u_data in zip(l_loader, u_loader):
    iteration += 1
    l_input, l_target = l_data
    l_input, l_target = l_input.to(device).float(), l_target.to(device).long()
    
    u_input, dummy_target, _ = u_data
    u_input, dummy_target = u_input.to(device).float(), dummy_target.to(device).long()
    
    
    target = torch.cat([l_target, dummy_target], 0)
    inputs = torch.cat([l_input, u_input], 0)

    if conf['set_cenmode'] == 'super':
        util.set_centroids(i=iteration,model=model,imgs=l_input,lbls=l_target)
    elif conf['set_cenmode'] == 'psed_super':
        p_input,p_target = util.filter_high_conf_psed(u_input,model,th=0.95)
        util.set_centroids(i=iteration,model=model,imgs=torch.cat([l_input,p_input]),lbls=torch.cat([l_target,p_target]))
    elif conf['set_cenmode'] == 'lcen':
        pass
    
    optimizer.zero_grad()
    outputs = model(inputs)

    #supervised_loss
    cls_loss = F.cross_entropy(outputs[1], target, reduction="none", ignore_index=-1).mean()
    
    #cm loss
    cm_loss  = cm_obj(outputs)

    if conf['alg'] != "supervised":   
        # ramp up exp(-5(1 - t)^2)  
        ramp_ssl  =  math.exp(-5 * (1 - min(iteration/conf["warmup"], 1))**2)
        ramp_cm   = 1
        
        coef_ssl  = conf['gamma'] * ramp_ssl
        coef_cm   = conf['beta']*ramp_cm

        unlabeled_mask = (target == -1).float()
        ssl_loss  =ssl_obj(inputs, outputs[1].detach(), model, unlabeled_mask)
        wssl_loss = ssl_loss * coef_ssl
        wcm_loss  = cm_loss * coef_cm

    else:
        ramp_cm  =  1
        coef_ssl  = 0
        coef_cm   = conf['beta']*ramp_cm

        ssl_loss = torch.zeros(1).to(device)   
        wssl_loss = coef_ssl*ssl_loss
        wcm_loss  = cm_loss * coef_cm
    
    loss = cls_loss + wssl_loss + wcm_loss

    loss.backward()
    optimizer.step()
    ema.update()


    if conf['alg'] == "MT" or conf['alg'] == "ICT":
        # parameter update with exponential moving average
        ssl_obj.moving_average(model.parameters())
    
    if iteration == 1 or (iteration % 5000) == 0:
        logger.log_scalars(data_dict={
                        'train/cls_loss' :cls_loss.item() , 
                        'train/CM_loss'  :cm_loss.item(),
                        'train/WCM_loss' :wcm_loss.item(),
                        'train/cm_coef'  :coef_cm,
                        'train/lr' :optimizer.param_groups[0]["lr"],
                        'train/loss': loss.item(),
                        'train/ssl_coef'  :coef_ssl,
                        'train/WSSL_loss' :wssl_loss.item() ,
                        'train/SSL_loss'  :ssl_loss.item()            
                        },it=iteration)
        
        
    # validation and test 
    if iteration == 1 or (iteration % args.validation) == 0 or (iteration == conf["iteration"]):
        acc,ema_acc,cert =  util.evaluate(loader=val_loader,model=model,ema=ema,device=device)
        test_acc,test_ema_acc,testcert =  util.evaluate(loader=test_loader,model=model,ema=ema,device=device)
        logger.log_scalars(data_dict={'test/val_acc':acc,'test/ema_val_acc':ema_acc,'test/val_cert':cert},it=iteration)
        logger.log_scalars(data_dict={'test/all_test_acc':test_acc,'test/all_ema_test_acc':test_ema_acc,'test/test_cert':testcert},it=iteration)
        
        if maximum_val_acc < acc:
            maximum_val_acc = acc
            logger.log_scalars(data_dict={'test/test_acc':test_acc},it=iteration)
            logger.log_scalars(data_dict={'test/ema_test_acc':test_ema_acc},it=iteration)
            if args.plot_umap:
                fig = util.get_UMAP_fig(loader=test_loader,model=model,device=device,colorlist=clist)
                logger.log_fig(fig_dict={'plot/umap':fig},it=iteration)


        rem_hrs = util.remain_hrs(p=iteration/conf["iteration"],st=st)
        logger.log_scalars(data_dict={'train/remain_hrs':rem_hrs},it=iteration)

    if iteration == 1 or (iteration % 5000) == 0:
        logger.print(it=iteration)
            
    # lr decay
    if iteration == conf["lr_decay_iter"]:
        optimizer.param_groups[0]["lr"] *= conf["lr_decay_factor"]