from lib.datasets import mnist
import numpy as np

config_pi_mnist = {

'mnist_pi_ce_100_config' : {
    "gamma":-0.70,
    "beta":'-inf',
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : mnist.MNIST,
    "num_classes" : 10,
    "nlabels": 100,
    "model": 'wr28-2',
    "alg": 'PI',
},


'mnist_pi_cecm_100_config' : {
    "gamma":-0.70,
    "beta":-0.7524,
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : mnist.MNIST,
    "num_classes" : 10,
    "nlabels": 100,
    "model": 'wr28-2',
    "alg": 'PI',
},


'mnist_pi_ce_1000_config' : {
    "gamma":-0.70,
    "beta":'-inf',
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : mnist.MNIST,
    "num_classes" : 10,
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'PI',
},


'mnist_pi_cecm_1000_config' : {
    "gamma":-0.70,
    "beta":-0.7524,
    "lr" :-3,
    "transform" : [False, True, False], # flip, rnd crop, gaussian noise
    "dataset" : mnist.MNIST,
    "num_classes" : 10,
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'PI',
},
}