import torch
import torch.nn as nn
import torch.nn.functional as F

class nt_mu(nn.Module):
    def __init__(self,indim,outdim):
        super(nt_mu, self).__init__()
        self.register_buffer('weight',torch.empty((outdim,indim), dtype=torch.float))
        nn.init.constant_(self.weight, 0)
        
    def forward(self,x):
        return F.linear(x,self.weight)


class cm(nn.Module):

    def __init__(self, input_dim, num_clusters,T=1):
        super(cm, self).__init__()
        self.input        = input_dim
        self.num_clusters = num_clusters
        self.register_buffer('Id',torch.eye(num_clusters))

        self.gamma_pre    = nn.Linear(input_dim, num_clusters)
        self.smax         = nn.Softmax(dim=-1)
        
        self.mu       = nt_mu(num_clusters, input_dim)
        self.T        = T   

    def _mu(self):
        return self.mu( self.Id )

        
    def forward(self, x):
        g_pre  = self.gamma_pre(x)/self.T
        g      = self.smax(g_pre)
        tx     = self.mu(g)
        u      = self.mu.weight.T
        return x, g_pre, g, tx, u



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def upconv3x3(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(out_planes, in_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation=F.relu):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.activation = activation

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out

class ResNet18_CM(nn.Module):
    def __init__(self,num_classes,transform_fn,T=1):
        super(ResNet18_CM, self).__init__()

        block           =  BasicBlock
        num_blocks      =  [2,2,2,2,2]
        activation      =  F.relu

        self.in_planes  = 64
        self.activation = activation
        
        self.backbone = nn.Sequential(
                                    conv3x3(3,64),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(.2),
                                    self._make_layer(block, 64, num_blocks[0], stride=1, activation=self.activation),
                                    self._make_layer(block, 128, num_blocks[1], stride=2, activation=self.activation),
                                    self._make_layer(block, 256, num_blocks[2], stride=2, activation=self.activation),
                                    self._make_layer(block, 512, num_blocks[3], stride=2, activation=self.activation),
                                    nn.AvgPool2d(4),
                                    nn.Flatten(),
                                    )


        self.OUT_FEATS_DIM = 512*block.expansion
        self.cm_module = cm(input_dim=self.OUT_FEATS_DIM, num_clusters=num_classes,T=T)
        self.transform_fn = transform_fn
       
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

    def _make_layer(self, block, planes, num_blocks, stride, activation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, activation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,return_feature=False):
        if self.training and self.transform_fn is not None:
            x = self.transform_fn(x)

        f = self.backbone(x)
        cm = self.cm_module(f.squeeze())

        if return_feature:
            return [cm, f.squeeze()]
        else:
            return cm

    def update_batch_stats(self, flag):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.update_batch_stats = flag

