import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from util import check_N_mkdir


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
def tsne_2D(embedded_tensor, labels, grps, figsize=(90, 90), rlt_dir=None, preffix='Weights', suffix=0):
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
    df = pd.DataFrame(zip(embedded_tensor[:, 0], embedded_tensor[:, 1], labels, grps))
    df.columns = ['coordX', 'coordY', 'labels', 'layers']
    df_deconv = df[df['layers'].str.contains('deconv|logit')]
    df_conv = df[~df['layers'].str.contains('deconv|logit')]

    # convert column groups to categories int
    df_conv['colors'] = pd.Categorical(df_conv['layers']).codes
    df_deconv['colors'] = pd.Categorical(df_deconv['layers']).codes
    df['colors'] = pd.Categorical(df['layers']).codes

    # 2D scatter plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    scat1 = ax1.scatter(df_conv['coordX'], df_conv['coordY'], c=df_conv['colors'], cmap='tab20', alpha=0.5)
    scat2 = ax2.scatter(df_deconv['coordX'], df_deconv['coordY'], c=df_deconv['colors'], cmap='tab20', alpha=0.5)
    scat3 = ax3.scatter(df['coordX'], df['coordY'], c=df['colors'], alpha=0.5)

    # set titles
    ax1.set_title('Conv layers weights')
    ax2.set_title('Deconv layers weights')
    ax3.set_title('All layers weights')

    # set legends
    leg1 = ax1.legend(scat1.legend_elements()[0], df_conv['layers'].unique(), title='Conv Layers')  #note: unique() might change order
    ax1.add_artist(leg1)
    leg2 = ax2.legend(scat2.legend_elements()[0], df_deconv['layers'].unique(), title='Deconv Layers')  #note: unique() might change order
    ax2.add_artist(leg2)
    leg3 = ax3.legend(scat3.legend_elements()[0], df['layers'].unique(), title='All Layers')  #note: unique() might change order
    ax3.add_artist(leg3)
    ax3.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))

    check_N_mkdir(rlt_dir)
    plt.savefig(rlt_dir + '{}_2D_plot_step{}.png'.format(preffix, suffix))
    plt.show()


def tsne_3D(embedded_tensor, labels, grps, figsize=(90, 90), rlt_dir=None, suffix=0):
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
    df.columns = ['coordX', 'coordY', 'coordZ', 'labels', 'layers']
    df_deconv = df[df['layers'].str.contains('deconv')]
    df_conv = df[~df['layers'].str.contains('deconv')]

    # convert colume groups to categories int
    df_conv['colors'] = pd.Categorical(df_conv['layers']).codes
    df_deconv['colors'] = pd.Categorical(df_deconv['layers']).codes
    df['colors'] = pd.Categorical(df['layers']).codes

    # plots conv
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)
    ax.set_title('Weights from encoder')
    ax.scatter(df_conv['coordX'], df_conv['coordY'], df_conv['coordZ'], c=df_conv['colors'], cmap=plt.get_cmap('Spectral'), marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    check_N_mkdir(rlt_dir)
    # plt.savefig(rlt_dir + 'conv_weights_3Dplot_step{}.png'.format(suffix))

    # plots deconv
    fig2 = plt.figure(figsize=figsize)
    ax2 = Axes3D(fig2)
    ax2.set_title('Weights from decoder')
    ax2.scatter(df_deconv['coordX'], df_deconv['coordY'], df_deconv['coordZ'], c=df_deconv['colors'], cmap=plt.get_cmap('Spectral'), marker='o')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # plots conv
    fig3 = plt.figure(figsize=figsize)
    ax3 = Axes3D(fig3)
    ax3.set_title('Weights from all layers')
    ax3.scatter(df['coordX'], df['coordY'], df['coordZ'], c=df['colors'], cmap=plt.get_cmap('Spectral'), marker='o')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    # plt.savefig(rlt_dir + 'conv_weights_3Dplot_step{}.png'.format(suffix))

    # plt.savefig(rlt_dir + 'deconv_weights_3Dplot_step{}.png'.format(suffix))
    plt.show()


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
        t_sne = TSNE(perplexity=perplexity, n_components=2, init='random', n_iter=niter)
        res = t_sne.fit_transform(tensor)
    elif mode == '3D':
        t_sne = TSNE(perplexity=perplexity, n_components=3, init='random', n_iter=niter)
        res = t_sne.fit_transform(tensor)
    else:
        raise ValueError('Please choose a mode among 2D, 3D, sklearn or tf!')
    return res

