from lib.datasets import cifar100
import numpy as np

config_vat_cifar100 = {

'cifar100_vat_ce_10000_config' : {
    "xi"          : 1e-6,
    "eps"         : 6,
    "gamma"       : -0.4075,
    "beta"        :'-inf',
    "lr"          : -3.445,
    "transform"   : [True, True, True],
    "dataset"     : cifar100.CIFAR100,
    "num_classes" : 100,    
    "nlabels"     : 10000,
    "model"       : 'wr28-8',
    "alg"         : 'VAT',
},

'cifar100_vat_cecm_10000_config' : {
    # virtual adversarial training
    "xi"          : 1e-6,
    "eps"         : 6,
    "gamma"       : 0.5674,
    "beta"        : -0.4201,
#   "lr"          : -2.5228,
    "lr"          : -3.445,
    "transform"   : [True, True, True],
    "dataset"     : cifar100.CIFAR100,
    "num_classes" : 100,    
    "nlabels"     : 10000,
    "model"       : 'wr28-8',
    "alg"         : 'VAT',
},

'cifar100_vat_ce_2500_config' : {
    "xi"          : 1e-6,
    "eps"         : 6,
    "gamma"       : -0.4075,
    "beta"        :'-inf',
    "lr"          : -3.445,
    "transform"   : [True, True, True],
    "dataset"     : cifar100.CIFAR100,
    "num_classes" : 100,    
    "nlabels"     : 2500,
    "model"       : 'wr28-8',
    "alg"         : 'VAT',
},



'cifar100_vat_cecm_2500_config' : {
    # virtual adversarial training
    "xi"          : 1e-6,
    "eps"         : 6,
    "gamma"       : 0.5674,
    "beta"        : -0.4201,
#   "lr"          : -2.5228,
    "lr"          : -3.445,
    "transform"   : [True, True, True],
    "dataset"     : cifar100.CIFAR100,
    "num_classes" : 100,    
    "nlabels"     : 2500,
    "model"       : 'wr28-8',
    "alg"         : 'VAT',
},


}