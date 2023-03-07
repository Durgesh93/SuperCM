from lib.datasets import svhn
import numpy as np

config_mt_svhn = {

'svhn_mt_ce_250_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma": -0.81,
    "beta":'-inf',
    "lr":  -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 250,
    "model": 'wr28-2',
    "alg":'MT',
},

'svhn_mt_cecm_250_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma":-0.81,
    "beta": -0.89,
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,
    "nlabels": 250,
    "model": 'wr28-2',
    "alg":'MT',
},


'svhn_mt_ce_1000_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma": -0.81,
    "beta":'-inf',
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg":'MT',
},


'svhn_mt_cecm_1000_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma":-0.81,
    "beta": -0.89,
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg":'MT',
},


}