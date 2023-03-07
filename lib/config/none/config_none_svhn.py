from lib.datasets import svhn
import numpy as np

config_none_svhn = {

'svhn_supervised_ce_250_config' : {
    "gamma":'-inf',
    "beta":'-inf',
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'supervised',
},


'svhn_supervised_ce_1k_config' : {
    "gamma":'-inf',
    "beta":'-inf',
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'svhn_supervised_cecm_250_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'supervised',
},


'svhn_supervised_cecm_1k_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'supervised',
},
}