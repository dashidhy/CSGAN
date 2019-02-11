import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#############
# functions
#############

def show_tensor_images(images, iter_count, figure_route):
    sqrtn = int(np.ceil(np.sqrt(images.shape[0])))

    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.transpose(1, 2, 0))
    plt.savefig(figure_route + str(iter_count - 1) + '.png')
    plt.close(fig)
    return

def show_loss(d_loss, g_loss, figure_route):
    
    fig = plt.figure()
    plt.plot(np.arange(len(d_loss)), d_loss)
    plt.xlabel('Iteration')
    plt.ylabel('D real Loss')
    plt.title('Discriminator training Loss')
    plt.savefig(figure_route + 'd_loss.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(np.arange(len(g_loss)), g_loss)
    plt.xlabel('Iteration')
    plt.ylabel('G Loss')
    plt.title('Generator training Loss')
    plt.savefig(figure_route + 'g_loss.png')
    plt.close(fig)
    return
