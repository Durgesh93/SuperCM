from lib.datasets import cifar10
import numpy as np

config_mt_cifar10 = {

'cifar10_mt_ce_600_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma": -0.81,
    "beta":'-inf',
    "lr":  -3,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 600,
    "model": 'wr28-2',
    "alg":'MT',
},

'cifar10_mt_cecm_600_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma":-0.81,
    "beta": -0.89,
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,
    "nlabels": 600,
    "model": 'wr28-2',
    "alg":'MT',
},


'cifar10_mt_ce_4000_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma": -0.81,
    "beta":'-inf',
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg":'MT',
},


'cifar10_mt_cecm_4000_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma":-0.81,
    "beta": -0.89,
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg":'MT',
},


}