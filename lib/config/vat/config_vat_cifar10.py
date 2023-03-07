from lib.datasets import cifar10
import numpy as np

config_vat_cifar10 = {

'cifar10_vat_ce_250_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma": -0.4075,
    "beta":'-inf',
    "lr"   : -3.445,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'VAT',
},


'cifar10_vat_ce_600_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma": -0.4075,
    "beta":'-inf',
    "lr"   : -3.445,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 600,
    "model": 'wr28-2',
    "alg": 'VAT',
},


'cifar10_vat_ce_1000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma": -0.4075,
    "beta":'-inf',
    "lr"   : -3.445,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'VAT',
},


'cifar10_vat_ce_2000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma": -0.4075,
    "beta":'-inf',
    "lr"   : -3.445,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 2000,
    "model": 'wr28-2',
    "alg": 'VAT',
},



'cifar10_vat_ce_4000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":0.1610,
    "beta":'-inf',
    "lr" : -2.5228,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg": 'VAT',
},




'cifar10_vat_cecm_250_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":0.75,
    "beta":-0.20,
    #"lr" : -2.5228,
    "lr" : -3.445,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 250,
    "model": 'wr28-2',
    "alg": 'VAT',
},



'cifar10_vat_cecm_600_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":0.75,
    "beta":-0.20,
    #"lr" : -2.5228,
    "lr" : -3.445,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 600,
    "model": 'wr28-2',
    "alg": 'VAT',
},


'cifar10_vat_cecm_1000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":0.75,
    "beta":-0.20,
    #"lr" : -2.5228,
    "lr" : -3.445,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'VAT',
},


'cifar10_vat_cecm_2000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":0.75,
    "beta":-0.20,
    #"lr" : -2.5228,
    "lr" : -3.445,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 2000,
    "model": 'wr28-2',
    "alg": 'VAT',
},



'cifar10_vat_cecm_4000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":0.5674,
    "beta":-0.4201,
    "lr" : -2.5228,
    "transform" : [True, True, True],
    "dataset" : cifar10.CIFAR10,
    "num_classes" : 10,    
    "nlabels": 4000,
    "model": 'wr28-2',
    "alg": 'VAT',
},
}
