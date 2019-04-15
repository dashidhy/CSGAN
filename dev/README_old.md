# Proposal for ECE285 project

A [PDF](https://github.com/dashidhy/styleGAN/blob/master/proposal/proposal.pdf) version available.


## Abstract

We are going to explore a new way that associate Generative Adversarial Networks (GANs) with style transfer techniques. We expect GANs with enhanced capacity can generate not only target contents, but also specific styles. Our goal is to learn a mapping G : z → I, where z is the domain of samples from low-dimensional latent variables, and I is the domain of target images. The distribution of images from G(z) should be indistinguishable both from the distribution of target content C and the distribution of specific style S.

## Background

A lot of brilliant works have been done on GANs and style transfer techniques, which have achieved many impressive results. One of the most amazing works is [CycleGAN proposed by Zhu et al.](https://arxiv.org/pdf/1703.10593.pdf), in which the authors implement a new structure of GANs to solve image-to-image translation problems. Our idea is inspired by CycleGAN, but what we will do is to come back to the origin of GANs — to generate target samples from low-dimensional latent variables. We expect our generate network G can create new images, not translate an image to another. This technique can be very useful in future artistic creation tasks, that just by manipulating low-dimensional parameters we can create artworks with desired contents and styles.

## Method

In our project, we are going to propose new strategies for training GANs to enhance their capacity in capturing the distributions of contents as well as styles. So far, we have a very simple idea that firstly train the GAN with content images, and then enhance it by training with a style loss network. This idea is inspired by [fast-neural-style technique proposed by Johnson et al.](https://arxiv.org/pdf/1603.08155.pdf), in which the authors use a CNN to approximate the style transfer mapping. In our work, we replace the CNN by GAN to achieve our goal. This change leads to different training strategy which we would like to describe it in detail in our final paper. We have done experiments that can generate promising images in 64x64 resolution. We are still working on the structure of our network and training strategy, and expect to get better results that generate higher resolution images.

We also have a more advanced idea that to improve our work by the idea from [InfoGAN proposed by Chen et al.](https://arxiv.org/pdf/1606.03657.pdf), which make our latent variable interpretable separately for contents and styles.

## Experiment results

We pretrain a [DCGAN](https://arxiv.org/pdf/1511.06434.pdf) on the church-outdoor class of [LSUN](http://lsun.cs.princeton.edu/2017/) dataset, and then continue training the generator on a image which contains target style. Output images shown below.

**Output from pretrained generator:**

<div align=center><img src="https://github.com/dashidhy/styleGAN/raw/master/images/pretrained_gan_output.png?sanitize=true"/></div>

</b>

**Target style:**

<div align=center><img src="https://github.com/dashidhy/styleGAN/raw/master/images/the_scream.jpg?sanitize=true"/  width="300"></div>

</b>

**Output from enhanced generator with lower learning rate:**

<div align=center><img src="https://github.com/dashidhy/styleGAN/raw/master/images/styleGAN_output.png?sanitize=true"/></div>

</b>

**Output from enhanced generator with higher learning rate:**

<div align=center><img src="https://github.com/dashidhy/styleGAN/raw/master/images/styleGAN_output2.png?sanitize=true"/></div>

