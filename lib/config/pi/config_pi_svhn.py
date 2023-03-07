from lib.datasets import svhn
import numpy as np

config_pi_svhn = {

'svhn_pi_ce_250_config' : {
    "gamma":-0.70,
    "beta":'-inf',
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'PI',
},


'svhn_pi_cecm_250_config' : {
    "gamma":-0.70,
    "beta":-0.7524,
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'PI',
},


'svhn_pi_ce_1000_config' : {
    "gamma":-0.70,
    "beta":'-inf',
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'PI',
},


'svhn_pi_cecm_1000_config' : {
    "gamma":-0.70,
    "beta":-0.7524,
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'PI',
},
}