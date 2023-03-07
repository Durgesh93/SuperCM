from lib.datasets import svhn
import numpy as np

config_vat_svhn = {

'svhn_vat_ce_250_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma": -0.4075,
    "beta":'-inf',
    "lr"   : -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'VAT',
},

'svhn_vat_cecm_250_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":-0.4075,
    "beta":-0.20,
    #"lr" : -2.5228,
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'VAT',
},


'svhn_vat_ce_1000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":-0.4075,
    "beta":'-inf',
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'VAT',
},


'svhn_vat_cecm_1000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":-0.4075,
    "beta":-0.20,
    #"lr" : -2.5228,
    "lr" : -3,
    "transform" : [True, True, True],
    "dataset" : svhn.SVHN,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'VAT',
},


}
