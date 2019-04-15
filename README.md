# StyleGAN: Large Scale Style Transfer Using Generative Adversarial Networks

This repository is built for the paper [StyleGAN: Large Scale Style Transfer Using Generative Adversarial Networks](https://github.com/dashidhy/styleGAN/blob/master/paper/StyleGAN.pdf). Please refer to it for more details.

<div align='center'>
  <img src='https://github.com/dashidhy/styleGAN/raw/master/images/candy_112.jpg?sanitize=true' height="350px">
  <img src='https://github.com/dashidhy/styleGAN/raw/master/images/muse_112.jpg?sanitize=true' height="350px">
  <img src='https://github.com/dashidhy/styleGAN/raw/master/images/mosaic_112.jpg?sanitize=true' height="350px">
  <img src='https://github.com/dashidhy/styleGAN/raw/master/images/udnie_112.jpg?sanitize=true' height="350px">
</div>

## Prerequisites

- Python3 (only tested on 3.6.4)
- Pytorch 0.4.0 (does not support other versions)
- NumPy (only tested on 1.14.2)
- Matplotlib (only tested on 2.2.0)
- Pillow (only tested on 5.1.0)
- Jupyter Notebook
- GPU support (a Nvidia GTX 1080 Ti is recommended)

## File Organization

- *datasets* ------ where you should set your datasets
- *dev* ------ some in-developing codes
- *images* ------ images for this README
- *models* ------ codes for building our models
- *paper* ------ a copy of our paper
- *savefigs* ------ HD images generated from demo notebooks will be saved here
- *savemodels* ------ two pretrained generators are provided here
- *styles* ------ images for style transfer
- *utils* ------ codes for data processing, visualization, and small modules
- *Generator_layer_understanding.ipynb* ------ notebook for reproducing our layer analysis
- *StyleGAN_training.ipynb* ------ notebook for reproducing our fine-tuning process
- *train_fixed_DCGAN.py* ------ script for training a 64x64 DCGAN
- *train_fixed_LSGAN.py* ------ script for training a 112x112 LSGAN

## Getting Started

### Installation

- Clone this repo:

```bash
git clone -b master --single-branch https://github.com/dashidhy/styleGAN.git
cd ./styleGAN
```

- Make sure you have all prerequisites ready

### Style Transfer

Pretrained models and Jupyter Notebooks are provided for our fine-tuning process. You can play with them without downloading datasets. If you would like to reproduce the whole work, please read the sections below.

### Prepare datasets

In our work, we use *church-outdoor* class of [LSUN](http://lsun.cs.princeton.edu/2017/) dataset for GAN training. Please refer to their [README](https://github.com/fyu/lsun/blob/master/README.md) file for downloading. Finally, you should have a folder named *church_outdoor_train_lmdb* under ./datasets/LSUN/. You can also implement your own datasets, but this may lead to some works on the source codes.

### Training a GAN

To train a 64x64 DCGAN, run:

```bash
python3 train_fixed_DCGAN.py -d
```

To train a 112x112 LSGAN, run:

```bash
python3 train_fixed_LSGAN.py -d
```

By using *-d* argument, you will run a training under our default settings. If you would like to set you own hyperparameters, use argument *-h* for help, or read and modify source codes directly.

