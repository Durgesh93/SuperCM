from lib.datasets import mnist
import numpy as np

config_vat_mnist = {

'mnist_vat_ce_100_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma": -0.4075,
    "beta":'-inf',
    "lr"   : -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 100,
    "model": 'wr28-2',
    "alg": 'VAT',
},

'mnist_vat_cecm_100_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":-0.4075,
    "beta":-0.20,
    #"lr" : -2.5228,
    "lr" : -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 100,
    "model": 'wr28-2',
    "alg": 'VAT',
},


'mnist_vat_ce_1000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":-0.4075,
    "beta":'-inf',
    "lr" : -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'VAT',
},


'mnist_vat_cecm_1000_config' : {
    # virtual adversarial training
    "xi"   : 1e-6,
    "eps"  : 6,
    "gamma":-0.4075,
    "beta":-0.20,
    #"lr" : -2.5228,
    "lr" : -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'VAT',
},


}
