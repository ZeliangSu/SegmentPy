import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# t-SNE on activation
def tsne_on_activation(embedded_tensor, labels, figsize=(45, 45), zoom=1, suffix='step0'):
    assert embedded_tensor.shape[0] >= len(labels), 'You should have embeddings then labels'
    fig, ax = plt.subplot(figsize=figsize)
    artists = []
    for xy, i in zip(embedded_tensor, labels):
        x, y = xy
        img = OffsetImage(i, zoom=zoom)
        ab = AnnotationBbox(img, (x, y), xycoords='data', framon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(embedded_tensor)
    ax.autoscale()
    plt.savefig(
        './dummy/tsne_act_{}.png'.format(suffix),  #fixme: change here
        dpi=45  # 2048 pixel divided by 45 = 45
    )


# t-SNE on kernel weights
def tsne_on_weights(embedded_tensor, labels, figsize=(90, 90), suffix='step0'):
    assert embedded_tensor.shape[0] >= len(labels), 'You should have more embeddings then labels'
    plt.figure(figsize=figsize)
    for i, label in enumerate(labels):
        x, y = embedded_tensor[i, ]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom'
        )
    plt.savefig(
        fname='./dummy/tsne_kernel{}.png'.format(suffix),  #fixme: change here
        dpi=50  # 2048 pixel divided by 45 = 45
    )

def tsne_on_weights_bis(embedded_tensor, labels, grps, figsize=(90, 90), suffix='step0'):
    assert embedded_tensor.shape[0] >= len(labels), 'You should have more embeddings then labels'
    plt.figure(figsize=figsize)
    for i, label in enumerate(labels):
        x, y = embedded_tensor[i, ]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom'
        )
    plt.savefig(
        fname='./dummy/tsne_kernel{}.png'.format(suffix),  #fixme: change here
        dpi=50  # 2048 pixel divided by 45 = 45
    )

# t-SNE on kernel weights
def tsne_on_weights_bis(embedded_tensor, labels, grps, figsize=(90, 90), suffix='step0'):
    assert embedded_tensor.shape[0] >= len(labels), 'You should have more embeddings then labels'
    plt.figure(figsize=figsize)
    df = pd.DataFrame(zip(embedded_tensor[:, 0], embedded_tensor[:, 1], labels, grps))
    df.columns = ['coordX', 'coordY', 'labels', 'groups']
    df_deconv = df[df['groups'].str.contains['deconv']]
    df_conv = df[~df['groups'].str.contains['deconv']]
    conv_plot = sns.lmplot(
        x='coordX',
        y='coordY',
        data=df_conv,
        fit_reg=False,
        hue='groups',
        legend=True,
        palette='Set1',
    )

    deconv_plot = sns.lmplot(
        x='coordX',
        y='coordY',
        data=df_deconv,
        fit_reg=False,
        hue='groups',
        legend=True,
        palette='Set1',
    )
    conv_plot.savefig(
        fname='./dummy/tsne_kernel{}_conv.png'.format(suffix),  #fixme: change here
        # dpi=50  # 2048 pixel divided by 45 = 45
    )
    deconv_plot.savefig(
        fname='./dummy/tsne_kernel{}_deconv.png'.format(suffix),  #fixme: change here
        # dpi=50  # 2048 pixel divided by 45 = 45
    )


def tsne_on_weights_pandas(embedded_tensor, labels, grps, figsize=(90, 90), suffix='step0'):
    assert embedded_tensor.shape[0] >= len(labels), 'You should have more embeddings then labels'
    plt.figure(figsize=figsize)
    for i, label in enumerate(labels):
        x, y = embedded_tensor[i, ]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom'
        )
    plt.savefig(
        fname='./dummy/tsne_kernel{}.png'.format(suffix),  #fixme: change here
        dpi=50  # 2048 pixel divided by 45 = 45
    )

def tsne_paper():
    # https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
    raise NotImplementedError('Please refer to Zeiler et al. 2014')


def tsne(tensor, perplexity=30, niter=5000, mode='2D'):
    if mode == '2D' or 'sklearn':
        t_sne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=niter)
        res = t_sne.fit_transform(tensor)
    elif mode == '3D' or 'tf':
        raise NotImplementedError('3D or tensorflow representation not implemented yet')
    else:
        raise ValueError('Please choose a mode among 2D, 3D, sklearn or tf!')
    return res
