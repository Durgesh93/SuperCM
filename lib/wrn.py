import torch
import torch.nn.functional as F
import torch.nn as nn

class nt_mu(nn.Module):
    def __init__(self,indim,outdim):
        super(nt_mu, self).__init__()
        self.register_buffer('weight',torch.empty((outdim,indim), dtype=torch.float))
        nn.init.constant_(self.weight, 0)
        
    def forward(self,x):
        return F.linear(x,self.weight)


class cm(nn.Module):

    def __init__(self, input_dim, num_clusters,T=1,setcen=True):
        super(cm, self).__init__()
        self.input        = input_dim
        self.num_clusters = num_clusters
        self.register_buffer('Id',torch.eye(num_clusters))

        self.gamma_pre    = nn.Linear(input_dim, num_clusters)
        self.smax         = nn.Softmax(dim=-1)
        if setcen:
            self.mu       = nt_mu(num_clusters, input_dim)
        else:
            self.mu       = nn.Linear(num_clusters, input_dim)
        self.T            = T   

    def _mu(self):
        return self.mu( self.Id )

        
    def forward(self, x):
        g_pre  = self.gamma_pre(x)/self.T
        g      = self.smax(g_pre)
        tx     = self.mu(g)
        u      = self.mu.weight.T
        return x, g_pre, g, tx, u



def conv3x3(i_c, o_c, stride=1):
    return nn.Conv2d(i_c, o_c, 3, stride, 1, bias=False)

class BatchNorm2d(nn.BatchNorm2d):
    def __init__(self, channels, momentum=1e-3, eps=1e-3):
        super().__init__(channels)
        self.update_batch_stats = True

    def forward(self, x):
        if self.update_batch_stats:
            return super().forward(x)
        else:
            return nn.functional.batch_norm(
                x, None, None, self.weight, self.bias, True, self.momentum, self.eps
            )

def relu():
    return nn.LeakyReLU(0.1)

class residual(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1, activate_before_residual=False):
        super().__init__()
        layer = []
        if activate_before_residual:
            self.pre_act = nn.Sequential(
                BatchNorm2d(input_channels),
                relu()
            )
        else:
            self.pre_act = nn.Identity()
            layer.append(BatchNorm2d(input_channels))
            layer.append(relu())
        layer.append(conv3x3(input_channels, output_channels, stride))
        layer.append(BatchNorm2d(output_channels))
        layer.append(relu())
        layer.append(conv3x3(output_channels, output_channels))

        if stride >= 2 or input_channels != output_channels:
            self.identity = nn.Conv2d(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.identity = nn.Identity()

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        x = self.pre_act(x)
        return self.identity(x) + self.layer(x)



class WRN_CM(nn.Module):
    """ WRN28-width with leaky relu (negative slope is 0.1)"""
    def __init__(self, width, num_classes, transform_fn=None,T=1,set_cenmode='super'):
        super().__init__()

        self.init_conv = conv3x3(3, 16)

        filters = [16, 16*width, 32*width, 64*width]

        unit1 = [residual(filters[0], filters[1], activate_before_residual=True)] + \
            [residual(filters[1], filters[1]) for _ in range(1, 4)]
        self.unit1 = nn.Sequential(*unit1)

        unit2 = [residual(filters[1], filters[2], 2)] + \
            [residual(filters[2], filters[2]) for _ in range(1, 4)]
        self.unit2 = nn.Sequential(*unit2)

        unit3 = [residual(filters[2], filters[3], 2)] + \
            [residual(filters[3], filters[3]) for _ in range(1, 4)]
        self.unit3 = nn.Sequential(*unit3)

        self.unit4 = nn.Sequential(*[BatchNorm2d(filters[3]), relu(), nn.AdaptiveAvgPool2d(1)])

        if set_cenmode == 'lcen':
            self.cm_module = cm(input_dim=filters[-1], num_clusters=num_classes,T=T,setcen=False)
        else:
            self.cm_module = cm(input_dim=filters[-1], num_clusters=num_classes,T=T,setcen=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.transform_fn = transform_fn

    def forward(self, x, return_feature=False):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)
        x = self.init_conv(x)
        x = self.unit1(x)
        x = self.unit2(x)
        x = self.unit3(x)
        f = self.unit4(x)
        cm = self.cm_module(f.squeeze())
        
        if return_feature:
            return [cm, f.squeeze()]
        else:
            return cm

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag






