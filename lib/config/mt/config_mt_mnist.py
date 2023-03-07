from lib.datasets import mnist
import numpy as np

config_mt_mnist = {

'mnist_mt_ce_100_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma": -0.81,
    "beta":'-inf',
    "lr":  -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 100,
    "model": 'wr28-2',
    "alg":'MT',
},

'mnist_mt_cecm_100_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma":-0.81,
    "beta": -0.89,
    "lr" : -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,
    "nlabels": 100,
    "model": 'wr28-2',
    "alg":'MT',
},


'mnist_mt_ce_1000_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma": -0.81,
    "beta":'-inf',
    "lr" : -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg":'MT',
},


'mnist_mt_cecm_1000_config' : {
    # mean teacher
    "ema_factor" : 0.95,
    "gamma":-0.81,
    "beta": -0.89,
    "lr" : -3,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,    
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg":'MT',
},


}