from lib.datasets import cifar10
import numpy as np

#-----------------------------------NONE---------------------------------------

config_none_cifar10 = {

'cifar10_supervised_ce_40_config' : {
    "gamma": '-inf',
    "beta":'-inf',
    "lr": -3.522,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 40,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'cifar10_supervised_ce_250_config' : {
    "gamma": '-inf',
    "beta":'-inf',
    "lr": -3.522,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'cifar10_supervised_ce_600_config' : {
    "gamma": '-inf',
    "beta":'-inf',
    "lr": -3.522,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 600,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'cifar10_supervised_ce_1k_config' : {
    "gamma": '-inf',
    "beta":'-inf',
    "lr": -3.522,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'cifar10_supervised_ce_4k_config' : {
    "gamma":'-inf',
    "beta":'-inf',
    "lr" :-3.5228,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'cifar10_supervised_ce_2k_config' : {
    "gamma":'-inf',
    "beta":'-inf',
    "lr" :-3.5228,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 2000,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'cifar10_supervised_cecm_40_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-3.52287,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 40,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'cifar10_supervised_cecm_250_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-3.52287,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'cifar10_supervised_cecm_600_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-3.52287,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 600,
    "model": 'wr28-2',
    "alg": 'supervised',
},


'cifar10_supervised_cecm_1k_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-3.52287,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'cifar10_supervised_cecm_2k_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-3.52287,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 2000,
    "model": 'wr28-2',
    "alg": 'supervised',
},


'cifar10_supervised_cecm_4k_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-3.52287,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg": 'supervised',
},

}