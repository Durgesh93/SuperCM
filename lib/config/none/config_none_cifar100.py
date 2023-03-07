from lib.datasets import cifar100
import numpy as np

config_none_cifar100 = {

'cifar100_supervised_ce_2500_config' : {
    "gamma":'-inf',
    "beta":'-inf',
    "lr" :-3,
    "transform" : [True, True, True],
    "dataset" : cifar100.CIFAR100,
    "num_classes" : 100,
    "nlabels": 2500,
    "model": 'wr28-8',
    "alg": 'supervised',
},

'cifar100_supervised_ce_10k_config' : {
    "gamma":'-inf',
    "beta":'-inf',
    "lr" :-3,
    "transform" : [True, True, True],
    "dataset" : cifar100.CIFAR100,
    "num_classes" : 100,
    "nlabels": 10000,
    "model": 'wr28-8',
    "alg": 'supervised',
},

'cifar100_supervised_cecm_2500_config' : {
    "gamma":'-inf',
    "beta":-1.8,
    "lr" :-3,
    "transform" : [True, True, True],
    "dataset" : cifar100.CIFAR100,
    "num_classes" : 100,
    "nlabels": 2500,
    "model": 'wr28-8',
    "alg": 'supervised',
},

'cifar100_supervised_cecm_10k_config' : {
    "gamma":'-inf',
    "beta":-1.8,
    "lr" :-3,
    "transform" : [True, True, True],
    "dataset" : cifar100.CIFAR100,
    "num_classes" : 100,
    "nlabels": 10000,
    "model": 'wr28-8',
    "alg": 'supervised',
},

}