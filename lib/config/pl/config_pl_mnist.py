from lib.datasets import mnist
import numpy as np

config_pl_mnist = {

'mnist_pl_ce_100_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma": -0.73,
    "beta":'-inf',
    "lr": -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 100,
    "model": 'wr28-2',
    "alg": 'PL',
},

'mnist_pl_cecm_100_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma":-0.73,
    "beta":0.48,
    "lr" :-3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 100,
    "model": 'wr28-2',
    "alg": 'PL',
},


'mnist_pl_ce_1000_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma":-0.73,
    "beta":'-inf',
    "lr" : -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'PL',
},


'mnist_pl_cecm_1000_config' : {
    # pseudo label
    "threashold" : 0.95,
    "gamma":-0.73,
    "beta":0.48,
    "lr" :-3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'PL',
},


}