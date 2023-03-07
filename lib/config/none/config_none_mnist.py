from lib.datasets import mnist
import numpy as np


config_none_mnist = {

'mnist_supervised_ce_100_config' : {
    "gamma":'-inf',
    "beta":'-inf',
    "lr" :-2.86,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,
    "nlabels": 100,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'mnist_supervised_ce_1000_config' : {
    "gamma":'-inf',
    "beta":'-inf',
    "lr" :-2.86,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'supervised',
},


'mnist_supervised_cecm_100_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-2.86,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,
    "nlabels": 100,
    "model": 'wr28-2',
    "alg": 'supervised',
},

'mnist_supervised_cecm_1000_config' : {
    "gamma":'-inf',
    "beta":-0.73,
    "lr" :-2.86,
    "transform" : [False, True, False],
    "dataset" : mnist.MNIST,
    "num_classes" : 10,
    "nlabels": 1000,
    "model": 'wr28-2',
    "alg": 'supervised',
},

}