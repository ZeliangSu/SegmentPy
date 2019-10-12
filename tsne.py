import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import seaborn as sns


# t-SNE on activation
def tsne_on_activation(embedded_tensor, labels, figsize=(45, 45), zoom=1, suffix='step0'):
    """
    inputs:
    -------
        embedded_tensor: (numpy ndarray)
        labels: (numpy ndarray?)
        figsize: (tuple of int)
        zoom: (int)
        suffix: (str)

    return:
    -------
        None
    """
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
    """
    inputs:
    -------
        embedded_tensor: (numpy ndarray)
        labels: (numpy ndarray?)
        figsize: (tuple of int)
        suffix: (str)

    return:
    -------
        None
    """
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
def tsne_on_weights_2D(embedded_tensor, labels, grps, figsize=(90, 90), rlt_dir=None, suffix='step0'):
    """
    inputs:
    -------
        embedded_tensor: (numpy ndarray)
        labels: (numpy ndarray?)
        grps: (pandas column)
        figsize: (tuple of int)
        suffix: (str)

    return:
    -------
        None
    """
    assert rlt_dir != None, "enter a rlt_dir"
    assert embedded_tensor.shape[0] >= len(labels), 'You should have more embeddings then labels'
    plt.figure(figsize=figsize)
    df = pd.DataFrame(zip(embedded_tensor[:, 0], embedded_tensor[:, 1], labels, grps))
    df.columns = ['coordX', 'coordY', 'labels', 'groups']
    df_deconv = df[df['groups'].str.contains('deconv')]
    df_conv = df[~df['groups'].str.contains('deconv')]
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
        fname=rlt_dir + 'tsne_{}_convKernels.png'.format(suffix),  #fixme: change here
        # dpi=50  # 2048 pixel divided by 45 = 45
    )
    deconv_plot.savefig(
        fname=rlt_dir + 'tsne_kernel{}_deconvKernels.png'.format(suffix),  #fixme: change here
        # dpi=50  # 2048 pixel divided by 45 = 45
    )


def tsne_on_weights_pandas(embedded_tensor, labels, figsize=(90, 90), rlt_dir=None, suffix='step0'):
    """
    inputs:
    -------
        embedded_tensor: (numpy ndarray) e.g. embedded_tensor.shape=[x, 2]
        labels: (numpy ndarray?)  e.g. labels.shape=[x]
        figsize: (tuple of int)
        suffix: (str)

    return:
    -------
    """
    assert rlt_dir != None, "enter a rlt_dir"
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
        fname=rlt_dir + 'tsne_{}.png'.format(suffix),  #fixme: change here
        dpi=50  # 2048 pixel divided by 45 = 45
    )


def tsne_on_weights_3D(embedded_tensor, labels, grps, figsize=(90, 90), rlt_dir=None, suffix='step0'):
    """
    inputs:
    -------
        embedded_tensor: (numpy ndarray)
        labels: (numpy ndarray?)
        grps: (pandas column)
        figsize: (tuple of int)
        suffix: (str)

    return:
    -------
        None
    """
    assert rlt_dir != None, "enter a rlt_dir"
    assert embedded_tensor.shape[0] >= len(labels), 'You should have more embeddings then labels'
    # group data with pandas
    df = pd.DataFrame(zip(embedded_tensor[:, 0], embedded_tensor[:, 1], embedded_tensor[:, 2], labels, grps))
    df.columns = ['coordX', 'coordY', 'coordZ', 'labels', 'groups']
    df_deconv = df[df['groups'].str.contains('deconv')]
    df_conv = df[~df['groups'].str.contains('deconv')]

    color_list = [i for i, _ in enumerate(df['group'])]
    # plots conv
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)
    ax.set_title('Weights from encoder')
    ax.scatter(df_conv['coordX'], df_conv['coordX'], df_conv['coordX'], c=df_conv['groups'], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.show()

    # plots deconv
    fig2 = plt.figure(figsize=figsize)
    ax2 = Axes3D(fig2)
    ax2.set_title('Weights from decoder')
    ax2.scatter(df_deconv['coordX'], df_deconv['coordX'], df_deconv['coordX'], c=df_deconv['groups'], marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    plt.show()


def tsne_paper():
    # https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
    raise NotImplementedError('Please refer to Zeiler et al. 2014')


def tsne(tensor, perplexity=30, niter=5000, mode='2D'):
    """
    inputs:
    -------
        tensor: (numpy ndarray)
        perplexity: (int)
        niter: (int)
        mode: (str)

    return:
    -------
        res: (numpy ndarray) reduced n-dimensions array
    """
    if mode == '2D':
        t_sne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=niter)
        res = t_sne.fit_transform(tensor)
    elif mode == '3D':
        t_sne = TSNE(perplexity=perplexity, n_components=3, init='pca', n_iter=niter)
        res = t_sne.fit_transform(tensor)
    else:
        raise ValueError('Please choose a mode among 2D, 3D, sklearn or tf!')
    return res
