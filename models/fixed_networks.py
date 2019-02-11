import torch.nn as nn
import utils.module_utils as mutils

#############
# functions
#############

def dc_Gen_BN_R():
    '''
    Fixed implementation of DCGAN generator
    Relu after Batchnorm
    Structure reference: https://arxiv.org/abs/1511.06434
    '''
    return nn.Sequential(
        nn.Linear(100, 1024 * 4 * 4),
        nn.BatchNorm1d(1024 * 4 * 4),
        nn.ReLU(inplace=True),
        mutils.Unflatten(-1, 1024, 4, 4),
        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.Tanh()
    )


def dc_Dis_BN_LR():
    '''
    Fixed implementation of DCGAN discriminator
    LeakyRelu after Batchnorm
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        mutils.Flatten(),
        nn.Linear(512 * 4 * 4, 1),
        nn.Sigmoid()
    )

    
def dc_Gen_R_BN():
    '''
    Fixed implementation of DCGAN generator
    Relu before Batchnorm
    Structure reference: https://arxiv.org/abs/1511.06434
    '''
    return nn.Sequential(
        nn.Linear(100, 1024 * 4 * 4),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024 * 4 * 4),
        mutils.Unflatten(-1, 1024, 4, 4),
        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(512),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.Tanh()
    )


def dc_Dis_LR_BN():
    '''
    Fixed implementation of DCGAN discriminator
    LeakyRelu before Batchnorm
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(256),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.BatchNorm2d(512),
        mutils.Flatten(),
        nn.Linear(512 * 4 * 4, 1),
        nn.Sigmoid()
    )


def dc_Gen_IN_R():
    '''
    Seems not work well
    '''
    return nn.Sequential(
        nn.Linear(100, 1024 * 4 * 4),
        nn.BatchNorm1d(1024 * 4 * 4),
        nn.ReLU(inplace=True),
        mutils.Unflatten(-1, 1024, 4, 4),
        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.InstanceNorm2d(512),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.InstanceNorm2d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=128, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1),
        nn.Tanh()
    )


def dc_Dis_IN_LR():
    '''
    Seems not work well
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.InstanceNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
        nn.InstanceNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        mutils.Flatten(),
        nn.Linear(512 * 4 * 4, 1),
        nn.Sigmoid()
    )


def lsv_Gen_BN_R():
    return nn.Sequential(
        nn.Linear(1024, 256 * 7 * 7),
        nn.BatchNorm1d(256 * 7 * 7),
        nn.ReLU(inplace=True),
        mutils.Unflatten(-1, 256, 7, 7),
        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
        nn.Tanh()
    )


def lsv_Dis_BN_LR():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        mutils.Flatten(),
        nn.Linear(512 * 7 * 7, 1),
        nn.Sigmoid()
    )


def ls_Gen_BN_R(ResGroup_size=2):
    model = [nn.Linear(1024, 256 * 8 * 8),
             nn.BatchNorm1d(256 * 8 * 8),
             nn.ReLU(inplace=True),
             mutils.Unflatten(-1, 256, 8, 8),
             nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
             nn.BatchNorm2d(256),
             nn.ReLU(inplace=True)]

    for i in range(ResGroup_size):
        model.append(mutils.ResBlock_trans(in_channels=256, out_channels=256))

    model += [nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1),
              nn.BatchNorm2d(256),
              nn.ReLU(inplace=True)]

    for i in range(ResGroup_size):
        model.append(mutils.ResBlock_trans(in_channels=256, out_channels=256))

    model += [nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
              nn.BatchNorm2d(128),
              nn.ReLU(inplace=True)]

    for i in range(ResGroup_size):
        model.append(mutils.ResBlock_trans(in_channels=128, out_channels=128))

    model += [nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
              nn.BatchNorm2d(64),
              nn.ReLU(inplace=True)]

    for i in range(ResGroup_size):
        model.append(mutils.ResBlock_trans(in_channels=64, out_channels=64))

    model += [nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
              nn.Tanh()]

    return nn.Sequential(*model)


def ls_Dis_BN_LR():
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        mutils.Flatten(),
        nn.Linear(512 * 8 * 8, 1),
        nn.Sigmoid()
    )