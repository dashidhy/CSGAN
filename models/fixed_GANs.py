import torch
from .baseGAN import BaseGAN
from . import fixed_networks as fnet
from torch.nn import init

##############
# classes
##############

class fixed_DCGAN(BaseGAN):

    def __init__(self, useGPU=True):
        super(fixed_DCGAN, self).__init__(64, 100, useGPU)


    def get_networks(self, net_type='BN_R'):
        '''
        Set generator and discriminator, default DCGAN with batchnorm
        '''
        if net_type == 'BN_R':
            self.G = fnet.dc_Gen_BN_R().type(self.type)
            self.D = fnet.dc_Dis_BN_LR().type(self.type)
        elif net_type == 'R_BN':
            self.G = fnet.dc_Gen_R_BN().type(self.type)
            self.D = fnet.dc_Dis_LR_BN().type(self.type)
        elif net_type == 'IN_R':
            self.G = fnet.dc_Gen_IN_R().type(self.type)
            self.D = fnet.dc_Dis_IN_LR().type(self.type)
        else:
            raise NotImplementedError('Net type [%s] is not implemented' % net_type)
        self.net_type = net_type


class fixed_LSGAN(BaseGAN):

    def __init__(self, useGPU=True):
        super(fixed_LSGAN, self).__init__(112, 1024, useGPU)

    def get_networks(self, net_type='BN_R'):

        if net_type == 'BN_R':
            self.G = fnet.lsv_Gen_BN_R().type(self.type)
            self.D = fnet.lsv_Dis_BN_LR().type(self.type)
        else:
            raise NotImplementedError('Net type [%s] is not implemented' % net_type)
        self.net_type = net_type


class LSGAN_Res(BaseGAN):

    def __init__(self, useGPU=True):
        super(LSGAN_Res, self).__init__(128, 1024, useGPU)


    def get_networks(self, net_type='BN_R', ResGroup_size=2):
        if net_type == 'BN_R':
            self.G = fnet.ls_Gen_BN_R(ResGroup_size).type(self.type)
            self.D = fnet.ls_Dis_BN_LR().type(self.type)
        else:
            raise NotImplementedError('Net type [%s] is not implemented' % net_type)
        self.net_type = net_type
        self.ResGroup_size = ResGroup_size


    def save_ckeckpoint(self, iter_count, epoch, model_route):
        checkpoint = {
                      'iter_count': iter_count,
                      'epoch': epoch,
                      'net_type': self.net_type,
                      'ResGroup_size': self.ResGroup_size,
                      'G_state_dict': self.G.state_dict(),
                      'D_state_dict': self.D.state_dict(),
                      'G_optim': self.G_optim,
                      'D_optim': self.D_optim,
                      'G_solver_state_dict': self.G_solver.state_dict(),
                      'D_solver_state_dict': self.D_solver.state_dict(),
                      'dset_name': self.dset_name,
                      'classes': self.classes,
                      'batch_size': self.batch_size,
                      'loss_name': self.loss_name,
                      'soft_label': self.soft_label
                     }
        torch.save(checkpoint, model_route + 'ckp_epoch_' + str(epoch) + '.pth')


    def load_model(self, ckp_route):
        
        ckp = torch.load(ckp_route)
        
        self.ckp_iter = ckp['iter_count']
        self.ckp_epoch = ckp['epoch']
        
        self.get_networks(net_type=ckp['net_type'], ResGroup_size=ckp['ResGroup_size'])
        self.G.load_state_dict(ckp['G_state_dict'])
        self.D.load_state_dict(ckp['D_state_dict'])
        
        self.get_G_optimizer(optim_name=ckp['G_optim'])
        self.get_D_optimizer(optim_name=ckp['D_optim'])
        self.G_solver.load_state_dict(ckp['G_solver_state_dict'])
        self.D_solver.load_state_dict(ckp['D_solver_state_dict'])
        
        self.get_dataset(dset_name=ckp['dset_name'], classes=[ckp['classes']])
        self.get_dataloader(ckp['batch_size'])
        self.get_loss(loss_name=ckp['loss_name'], soft_label=ckp['soft_label'])


    def init_weights(self, init_type='kaiming', gain=0.02):
        
        def init_func_G(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        def init_func_D(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('Initialize network with %s' % init_type)
        self.G.apply(init_func_G)
        self.D.apply(init_func_D)