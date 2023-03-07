from lib.datasets import svhn
import numpy as np

config_pl_svhn = {

'svhn_pl_ce_250_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma": -0.73,
    "beta":'-inf',
    "lr": -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'PL',
},

'svhn_pl_cecm_250_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma":-0.73,
    "beta":0.48,
    "lr" :-3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'PL',
},


'svhn_pl_ce_1000_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma":-0.73,
    "beta":'-inf',
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'PL',
},


'svhn_pl_cecm_1000_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma":-0.73,
    "beta":0.48,
    "lr" :-3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'PL',
},


}