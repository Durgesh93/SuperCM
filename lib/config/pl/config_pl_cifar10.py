from lib.datasets import cifar10
import numpy as np

config_pl_cifar10 = {

'cifar10_pl_cecm_4000_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma":-0.7332,
    "beta":0.4827,
    "lr" :-3.5228,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg": 'PL',
},

'cifar10_pl_ce_4000_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma":-0.8751,
    "beta":'-inf',
    "lr" : -3.5228,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg": 'PL',
},

'cifar10_pl_ce_600_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma": -0.8751,
    "beta":'-inf',
    "lr": -2.938,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 600,
    "model": 'wr28-2',
    "alg": 'PL',
},

'cifar10_pl_cecm_600_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma":-0.7332,
    "beta":0.4827,
    "lr" :-3.5228,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 600,
    "model": 'wr28-2',
    "alg": 'PL',
},

}