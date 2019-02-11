import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset

import os
import json

from .losses import G_Label_Loss, D_Label_Loss
import utils.gan_utils as gutils
import utils.vis_utils as vutils

##############
# classes
##############

class BaseGAN():

    def __init__(self, image_size, noise_dim, useGPU=True):

        # set model to GPU or CPU, default GPU
        if useGPU and torch.cuda.is_available():
            self.type = torch.cuda.FloatTensor
        else:
            self.type = torch.FloatTensor
            print('WARNING! GPU not used or not available, model set to CPU!')

        # constants
        self.image_size = image_size
        self.noise_dim = noise_dim
        
        # placeholder
        self.G = None
        self.D = None
        self.dataset = None
        self.dataloader = None
        self.batch_size = None
        self.G_solver = None
        self.D_solver = None
        self.G_loss = None
        self.D_loss = None
        self.ckp_iter = 0
        self.ckp_epoch = 0

        # information
        self.net_type = None
        self.dset_name = None
        self.classes = None
        self.G_optim = None
        self.D_optim = None
        self.loss_name = None
        self.soft_label = None

        # flags
        self.isinit = False

    def get_networks(self):
        pass


    def get_dataset(self, dset_name='LSUN', classes=['church_outdoor_train']):
        '''
        Set dataset, default LSUN church_outdoor_train, need download first.
        '''
        if dset_name == 'LSUN':
            self.dataset = dset.LSUN('./datasets/LSUN', classes=classes, 
                                     transform=gutils.rescale_training_set(self.image_size))
        else:
            raise NotImplementedError('Dataset [%s] is not implemented' % dset_name)
        self.dset_name = dset_name
        self.classes = classes[0]


    def get_dataloader(self, batch_size=128):
        '''
        Set dataloader, default batch size 128, shuffle.
        '''
        if self.dataset == None:
            raise NotImplementedError('Dataset is not implemented')
        else:
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
            self.batch_size = batch_size


    def get_G_optimizer(self, optim_name='Adam', lr=2e-4, betas=(0.5, 0.999)):
        '''
        Set generator optimizer, default Adam with lr=2e-4, betas=(0.5, 0.999).
        '''
        if optim_name == 'Adam':
            self.G_solver = optim.Adam(self.G.parameters(), lr=lr, betas=betas)
        else:
            raise NotImplementedError('Optimizer [%s] is not implemented' % optim_name)
        self.G_optim = optim_name


    def get_D_optimizer(self, optim_name='Adam', lr=2e-4, betas=(0.5, 0.999)):
        '''
        Set discriminator optimizer, default Adam with lr=2e-4, betas=(0.5, 0.999).
        '''
        if optim_name == 'Adam':
            self.D_solver = optim.Adam(self.D.parameters(), lr=lr, betas=betas)
        else:
            raise NotImplementedError('Optimizer [%s] is not implemented' % optim_name)
        self.D_optim = optim_name


    def get_loss(self, loss_name='LSGAN', soft_label=True):
        '''
        Set objective, default least square loss with soft label.
        '''
        self.G_loss = G_Label_Loss(loss_name=loss_name, soft_label=soft_label)
        self.D_loss = D_Label_Loss(loss_name=loss_name, soft_label=soft_label)
        self.loss_name = loss_name
        self.soft_label = soft_label


    def initialize(self):
        '''
        Check implementation of modules, if not implemented, use default.
        '''

        # test networks
        if self.net_type == None:
            self.get_networks()
            print('Using default networks.')
        else:
            print('Using self-defined or checkpoint networks.')

        # test dataset
        if self.dataset == None:
            self.get_dataset()
            print('Using default dataset. LSUN church_outdoor_train.')
        else:
            print('Using self-defined or checkpoint dataset.')

        # test dataloader
        if self.dataloader == None:
            self.get_dataloader()
            print('Using default dataloader. batch_size=128, shuffle=True.')
        else:
            print('Using self-defined or checkpoint dataloader.')

        # test optimizer
        if self.G_solver == None:
            self.get_G_optimizer()
            print('Using default generator optimizer. Adam with lr=2e-4, betas=(0.5, 0.999).')
        else:
            print('Using self-defined or checkpoint generator optimizer.')

        if self.D_solver == None:
            self.get_D_optimizer()
            print('Using default discriminator optimizer. Adam with lr=2e-4, betas=(0.5, 0.999).')
        else:
            print('Using self-defined or checkpoint discriminator optimizer.')

        # test loss
        if self.loss_name == None:
            self.get_loss()
            print('Using default loss. LSGAN with soft label.')
        else:
            print('Using self-defined or checkpoint loss.')

        self.isinit = True


    def lr_decay(self, G_decay=0.9, D_decay=0.85):
        '''
        Implement learning rate decay.
        '''
        for param_group in self.G_solver.param_groups:
            param_group['lr'] *= G_decay
        for param_group in self.D_solver.param_groups:
            param_group['lr'] *= D_decay


    def forward_G(self):
        fake_seed = gutils.input_noise_uniform(self.batch_size, self.noise_dim).type(self.type)
        return self.G(fake_seed)


    def iter_D(self, real_data, fake_images, th=0.2):
        self.D_solver.zero_grad()
        logits_real = self.D(real_data)
        logits_fake = self.D(fake_images)
        error = self.D_loss(logits_real=logits_real, logits_fake=logits_fake)
        loss = error.item()
        if loss > th:
            error.backward()
            self.D_solver.step()
        return loss


    def iter_G(self):
        self.G_solver.zero_grad()
        fake_images = self.forward_G()
        logits = self.D(fake_images)
        error = self.G_loss(logits=logits)
        error.backward()
        self.G_solver.step()
        return error.item()


    def iter_GAN(self, x, threshold_D):

        # train D
        real_data = gutils.preprocess_img(x.type(self.type))
        fake_images = self.forward_G()
        d_loss = self.iter_D(real_data, fake_images.clone().detach(), th=threshold_D)

        # train G with new input
        g_loss = self.iter_G()

        return d_loss, g_loss, fake_images


    def save_loss(self, d_loss, g_loss, epoch, model_route):
        route_d = model_route + 'd_loss_' + str(epoch) + '.txt'
        route_g = model_route + 'g_loss_' + str(epoch) + '.txt'
        os.mknod(route_d)
        os.mknod(route_g)
        with open(route_d, 'w') as fp:
            fp.write(json.dumps(d_loss))
        with open(route_g, 'w') as fp:
            fp.write(json.dumps(g_loss))


    def save_ckeckpoint(self, iter_count, epoch, model_route):
        checkpoint = {
                      'iter_count': iter_count,
                      'epoch': epoch,
                      'net_type': self.net_type,
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
        
        self.get_networks(net_type=ckp['net_type'])
        self.G.load_state_dict(ckp['G_state_dict'])
        self.D.load_state_dict(ckp['D_state_dict'])
        
        self.get_G_optimizer(optim_name=ckp['G_optim'])
        self.get_D_optimizer(optim_name=ckp['D_optim'])
        self.G_solver.load_state_dict(ckp['G_solver_state_dict'])
        self.D_solver.load_state_dict(ckp['D_solver_state_dict'])
        
        self.get_dataset(dset_name=ckp['dset_name'], classes=[ckp['classes']])
        self.get_dataloader(ckp['batch_size'])
        self.get_loss(loss_name=ckp['loss_name'], soft_label=ckp['soft_label'])


    def _check_route(self, model_route, figure_route):

        if model_route[-1] != '/':
            model_route += '/'

        if figure_route[-1] != '/':
            figure_route += '/'

        if os.path.exists(model_route):
            print('WARNING! Model route already exists, may cover previous models!')
        else:
            os.mkdir(model_route)
            print('Model route ' + model_route + ' created.')

        if os.path.exists(figure_route):
            print('WARNING! Figure route already exists, may cover previous figures!')
        else:
            os.mkdir(figure_route)
            print('Figure route ' + figure_route + ' created.')

        return model_route, figure_route


    def train(self, ckp_route=None, num_epochs=8, show_every=250, threshold_D=0.2,
              lr_decay_every=1000, G_decay=0.9, D_decay=0.85,
              model_route='./savemodels/', figure_route='./savefigs/'):
        
        model_route, figure_route = self._check_route(model_route, figure_route)
        
        # load checkpoint if given route, else train from zero
        if ckp_route is not None:
            print('Continue training from checkpoint ' + ckp_route)
            self.load_model(ckp_route)
            print('Ckeckpoint loaded successfully.')
        else:
            print('Train from zero.')

        # check settings, use default if something is not set
        if not self.isinit:
            self.initialize()
        else:
            print('WARNING! Model has been initialized or may has been trained!')

        iter_count = self.ckp_iter + 1
        d_loss = []
        g_loss = []

        # Start training
        for epoch in range(self.ckp_epoch, self.ckp_epoch + num_epochs):
            for x, _ in self.dataloader:

                if len(x) != self.batch_size:
                    continue

                d_error, g_error, fake_images = self.iter_GAN(x, threshold_D=threshold_D)
                d_loss.append(d_error)
                g_loss.append(g_error)

                if iter_count % show_every == 1:
                    imgs_numpy = gutils.deprocess_img(fake_images.detach()).cpu().numpy()

                    # show/save images and loss
                    vutils.show_tensor_images(imgs_numpy[0:64], iter_count, figure_route)
                    vutils.show_loss(d_loss, g_loss, figure_route)

                if iter_count % lr_decay_every == 0:
                    self.lr_decay(G_decay, D_decay)

                iter_count += 1

            # save model and loss per epoch
            self.save_ckeckpoint(iter_count - 1, epoch + 1, model_route)
            self.save_loss(d_loss, g_loss, epoch + 1, model_route)