import torch
import torchvision.transforms as trans

##############
# functions
##############

def preprocess_img(x):
    '''
    Rescale an image with pixel value in [0, 1]
    to [-1, 1]
    '''
    return 2.0 * (x - 0.5)

def deprocess_img(x):
    '''
    Rescale an image with pixel value in [-1, 1]
    to [0, 1]
    '''
    return (x + 1.0) / 2.0

def input_noise_uniform(batch_size, noise_dim):
    '''
    Uniform input noise from U[-1, 1]
    '''
    return 2.0 * (torch.rand(batch_size, noise_dim) - 0.5)

def input_noise_gaussian(batch_size, noise_dim):
    '''
    Gaussian input noise
    '''
    pass

def rescale_training_set(h):
    '''
    resize the training image to h by h center crops
    '''
    return trans.Compose([trans.Resize(h),
                          trans.CenterCrop(h),
                          trans.ToTensor()
                         ])
