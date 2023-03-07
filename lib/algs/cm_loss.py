import torch
import torch.nn as nn
import torch.nn.functional as F

class CM_loss(nn.Module):
    def __init__(self, num_clusters, alpha=1, lbd=0, orth=False, normalize=True):
        super(CM_loss, self).__init__()
        
        if hasattr( alpha, '__iter__'):
            self.register_buffer('alpha',
                                 torch.tensor(alpha))
        else:
            self.register_buffer(
                                 'alpha',
                                  torch.ones( num_clusters) * alpha
                                )
    
        self.lbd   = lbd
        self.orth = orth
        self.normalize = normalize
        self.register_buffer('Id',torch.eye(num_clusters))
        self.register_buffer('mask',1-torch.eye(num_clusters))
        

    def forward(self, inputs, targets=None, split=False):
        x,_,g,tx,u = inputs
        n,d = x.shape
        k = g.shape[1]
        
        nd = (n*d) if self.normalize else 1.
        
        loss_E1 = torch.sum( torch.square( x - tx ) ) / nd
        
        if self.orth:
            loss_E2 = torch.sum( g*(1-g) ) / nd
            uu = torch.matmul( u, u.T )
            loss_E3 = torch.sum( torch.square( uu - self.Id.to(uu.device) ) ) * self.lbd
        else:
            loss_E2 = torch.sum( torch.sum( g*(1-g),0 ) * torch.sum( torch.square(u), 1) ) / nd
            gg = torch.matmul( g.T, g)
            uu = torch.matmul( u, u.T )
            gu = gg * uu
            gu = gu * self.mask
            loss_E3 = - torch.sum( gu ) / nd
            
        
        lmg = torch.log( torch.mean(g,0) +1e-10 )
        loss_E4 = lmg
        
        if split:
            nd  = 1. if self.normalize else n*d
            return torch.stack( (loss_E1/nd, loss_E2/nd, loss_E3/(1 if self.orth else nd) , torch.sum(loss_E4* (1-self.alpha)) ) )
        else:
            return loss_E1 + loss_E2 + loss_E3 + torch.sum( loss_E4 * (1-self.alpha) )
            #return loss_E1