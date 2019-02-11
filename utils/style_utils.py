import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torchvision.transforms as T
import torch.optim as optim
import PIL

import numpy as np

from .gan_utils import deprocess_img

SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def style_target_preprocess(style_target, size=512, batchsize=128):
    transform = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=SQUEEZENET_MEAN.tolist(),
                    std=SQUEEZENET_STD.tolist()),
        T.Lambda(lambda x: x.repeat(batchsize, 1, 1, 1)),
    ])
    return transform(style_target)

def fake_image_preprocess(fake_image):
    mean = torch.tensor(SQUEEZENET_MEAN.reshape((1, 3, 1, 1))).type(fake_image.type())
    std = torch.tensor(SQUEEZENET_STD.reshape((1, 3, 1, 1))).type(fake_image.type())
    return (deprocess_img(fake_image) - mean) / std

def deprocess(img):
    transform = T.Compose([
        T.Lambda(lambda x: x[0]),
        T.Normalize(mean=[0, 0, 0], std=[1.0 / s for s in SQUEEZENET_STD.tolist()]),
        T.Normalize(mean=[-m for m in SQUEEZENET_MEAN.tolist()], std=[1, 1, 1]),
        T.Lambda(rescale),
        T.ToPILImage(),
    ])
    return transform(img)

def rescale(x):
    low, high = x.min(), x.max()
    x_rescaled = (x - low) / (high - low)
    return x_rescaled

def extract_features(x, cnn):
    features = []
    prev_feat = x
    for i, module in enumerate(cnn._modules.values()):
        next_feat = module(prev_feat)
        features.append(next_feat)
        prev_feat = next_feat
    return features

def content_loss(content_weight, content_current, content_original, batchsize=128):
    return content_weight * torch.sum((content_current - content_original)**2) / batchsize

def gram_matrix(features, normalize=True):
    N, C, H, W = features.shape
    feature_flatten = features.view([N, C, H*W])
    gram = torch.bmm(feature_flatten, feature_flatten.transpose(1, 2))
    if normalize:
        gram /= H * W * C
    return gram

def style_loss(feats, style_layers, style_targets, style_weights, batchsize=128):
    loss = 0.0
    for i in range(len(style_layers)):
        loss += style_weights[i] * torch.sum((gram_matrix(feats[style_layers[i]]) - style_targets[i])**2)
    return loss / batchsize

def tv_loss(img, tv_weight, batchsize=128):
    return tv_weight * (torch.sum((img[..., :-1, :] - img[..., 1:, :])**2) + torch.sum((img[..., :-1] - img[..., 1:])**2)) / batchsize  