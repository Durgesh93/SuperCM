         

from .none.config_none_cifar10  import config_none_cifar10            
from .none.config_none_cifar100 import config_none_cifar100             
from .none.config_none_mnist    import config_none_mnist          
from .none.config_none_svhn     import config_none_svhn         

from .mt.config_mt_cifar10      import config_mt_cifar10
from .mt.config_mt_svhn         import config_mt_svhn
from .mt.config_mt_mnist        import config_mt_mnist

from .pi.config_pi_cifar10      import config_pi_cifar10
from .pi.config_pi_svhn         import config_pi_svhn
from .pi.config_pi_mnist        import config_pi_mnist

from .pl.config_pl_cifar10      import config_pl_cifar10            
from .pl.config_pl_svhn         import config_pl_svhn
from .pl.config_pl_mnist        import config_pl_mnist

from .vat.config_vat_cifar10    import config_vat_cifar10     
from .vat.config_vat_svhn       import config_vat_svhn      
from .vat.config_vat_cifar100   import config_vat_cifar100            
from .vat.config_vat_mnist      import config_vat_mnist



config_dict_list = [
    
    config_none_cifar10,
    config_none_cifar100,
    config_none_mnist,
    config_none_svhn,


    config_pi_cifar10,
    config_pi_svhn,
    config_pi_mnist,
    
    config_pl_cifar10,
    config_pl_svhn,
    config_pl_mnist,    

    config_mt_cifar10,
    config_mt_svhn,
    config_mt_mnist,
    

    config_vat_cifar10,
    config_vat_cifar100,
    config_vat_svhn,
    config_vat_mnist,

]


shared_config = {
    "iteration" : 500000,
    "warmup" : 200000,
    "lr_decay_iter" : 400000,
    "lr_decay_factor" : 0.2,
    "batch_size" : 100,
    "T":1,
    "set_cenmode":"super",
}




### Settings ###


def get_config(key):
    setting = {}
    for dict in config_dict_list:
        setting.update(dict)
    setting[key].update(shared_config)
    return setting[key]

def update_args(setting,args):
    setting['gamma']       = args.gamma
    setting['beta']        = args.beta
    setting['lr']          = args.lr
    setting['T']           = args.T
    setting['set_cenmode'] = args.set_cenmode
    return setting