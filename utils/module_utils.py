import torch.nn as nn

##############
# functions
##############

def ConvTranspose3x3(in_planes, out_planes, stride=1):
    
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


##############
# classes
##############

class Flatten(nn.Module):
    '''
    Flatten layer for conv to fc
    '''
    def forward(self, x):
        N, _, _, _, = x.size()
        return x.view(N, -1)


class Unflatten(nn.Module):
    '''
    Unflatten layer for fc to conv
    '''
    def __init__(self, N, C, H, W):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


class ResBlock_trans(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock_trans, self).__init__()
        self.ConvTrans1 = ConvTranspose3x3(in_channels, out_channels, stride)
        self.BN1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.ConvTrans2 = ConvTranspose3x3(out_channels, out_channels)
        self.BN2 = nn.BatchNorm2d(out_channels)
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.ConvTrans1(x)
        out = self.BN1(out)
        out = self.ReLU(out)
        out = self.ConvTrans2(out)
        out = self.BN2(out)
        out += residual
        out = self.ReLU(out)

        return out