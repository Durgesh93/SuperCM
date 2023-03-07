from lib.datasets import cifar10
import numpy as np

config_pi_cifar10 = {

'cifar10_pi_ce_600_config' : {
    # Pi Model
    "gamma": -0.70,
    "beta":'-inf',
    "lr": -3,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 600,
    "model": 'wr28-2',
    "alg": 'PI',
},


'cifar10_pi_cecm_600_config' : {
    # Pi Model
    "gamma":-0.70,
    "beta":-0.7524,
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 600,
    "model": 'wr28-2',
    "alg": 'PI',
},

'cifar10_pi_ce_4000_config' : {
    # Pi Model
    "gamma":-0.70,
    "beta":'-inf',
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg": 'PI',
},


'cifar10_pi_cecm_4000_config' : {
    # Pi Model
    "gamma":-0.70,
    "beta":-0.7524,
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg": 'PI',
},

}