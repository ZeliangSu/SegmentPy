import tensorflow as tf
import numpy as np
import pandas as pd
import os
from util import get_all_trainable_variables
from tsne import tsne, tsne_on_weights_2D, tsne_on_weights_3D
from visualize import *


def tsne_partialRes_weights(params=None, conserve_nodes=None, mode='2D'):
    """
    input:
    -------
        ckptpath: (string) path to the checkpoint that we convert to .pb. e.g. './logs/YYYY_MM_DD_.../hourHH/ckpt/step{}'
    return:
    -------
        None
    """
    assert params!=None, 'please define the dictionary of paths'
    assert conserve_nodes!=None, 'please define the list of nodes that you conserve'
    assert isinstance(params, dict), 'paths should be a dictionary containning path'
    assert isinstance(conserve_nodes, list), 'conoserve_nodes should be a list of node names'
    # run tsne on wieghts
    # get weights from checkpoint
    wns, _, ws, _, _, _, _, _ = get_all_trainable_variables(params['ckpt_path'])

    # arange label and kernel
    new_wn = []
    new_ws = []
    grps = []
    for wn, w in zip(wns, ws):  # w.shape = [c_w, c_h, c_in, nb_conv]
        for i in range(w.shape[3]):
            new_wn.append(wn + '_{}'.format(i))  # e.g. conv4bis_96
            grps.append(wn.split('/')[0])

            #note: associativity: a x b + a x c = a x (b + c)
            # "...a kernel is the sum of all the dimensions in the previous layer..."
            # https://stackoverflow.com/questions/42712219/dimensions-in-convolutional-neural-network
            new_ws.append(np.sum(w[:, :, :, i], axis=2))  # e.g. (3, 3, 12, 24) [w, h, in, nb_conv] --> (3, 3, 24)

    # inject into t-SNE
    res = tsne(np.array(new_ws).transpose((1, 2, 0)).reshape(len(new_ws), -1),
               perplexity=params['perplexity'],
               niter=params['niter'], mode=mode)  # e.g. (3, 3, x) --> (9, x) --> (x, 2) or (x, 3)

    # mkdir

    # visualize the tsne
    if mode == '2D':
        tsne_on_weights_2D(res, new_wn, grps, rlt_dir=params['tsne_dir'], suffix=params['step'])
    elif mode == '3D':
        tsne_on_weights_3D(res, new_wn, grps, rlt_dir=params['tsne_dir'], suffix=params['step'])
    else:
        raise NotImplementedError('please choose 2D or 3D mode')


def weights_hists_2excel(ckpt_dir=None):
    """
    inputs:
    -------
        path: (string) path to get the checkpoint e.g. './logs/YYYY_MM_DD_.../hourHH/ckpt/'
    return:
    -------
        None
    """

    # note
    # construct dataframe
    # header sheet_name conv1: [step0, step20, ...]
    # header sheet_name conv1bis: [step0, step20, ...]

    #construct list [step0, step100, step200...]
    #ckpt name convention: step{}.meta
    lnames = []
    for step in os.listdir(ckpt_dir):
        if step.endswith('.meta'):
            lnames.append(step.split('.')[0])

    # add histograms and save in excel
    for step in lnames:
        # get weights-bias names and values
        wn, bn, ws, bs, dnn_wn, dnn_bn, dnn_ws, dnn_bs = get_all_trainable_variables(ckpt_dir + step)

        # construct a dict of key: layer_name, value: flattened kernels
        dfs = {sheet_name.split(':')[0].replace('/', '_'): pd.DataFrame({step: params.flatten()})
               for sheet_name, params in zip(wn + bn + dnn_wn + dnn_bn, ws + bs + dnn_ws + dnn_bs)}

        # write into excel
        with pd.ExcelWriter('./result/params.xlsx', engine='xlsxwriter') as writer:
            for sheet_name in dfs.keys():
                dfs[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)


def partialRlt_and_diff(paths=None, conserve_nodes=None):
    """
    input:
    -------
        paths: (dict) paths of the checkpoint that we convert to .pb. e.g. './logs/YYYY_MM_DD_.../hourHH/ckpt/step{}'
    return:
    -------
        None
    """
    assert paths!=None, 'please define the dictionary of paths'
    assert conserve_nodes!=None, 'please define the list of nodes that you conserve'
    assert isinstance(paths, dict), 'paths should be a dictionary containning path'
    assert isinstance(conserve_nodes, list), 'conoserve_nodes should be a list of node names'

    # load ckpt
    patch_size = int(paths['data_dir'].split('/')[-2])

    try:
        batch_size = int(paths['working_dir'].split('bs')[1].split('_')[0])
    except:
        batch_size = 100

    new_ph = tf.placeholder(tf.float32, shape=[batch_size, patch_size, patch_size, 1], name='new_ph')

    # define nodes to conserve

    # convert ckpt to pb
    convert_ckpt2pb(input=new_ph, paths=paths, conserve_nodes=conserve_nodes)

    # load main graph
    g_main, ops_dict = load_mainGraph(conserve_nodes, path=paths['save_pb_path'])

    # run nodes and save results
    inference_and_save_partial_res(g_main, ops_dict, conserve_nodes, input_dir=paths['data_dir'], rlt_dir=paths['rlt_dir'])


if __name__ == '__main__':
    # Xlearn
    # conserve_nodes = [
    #     'model/encoder/conv1/relu',
    #     'model/encoder/conv1bis/relu',
    #     'model/encoder/conv2/relu',
    #     'model/encoder/conv2bis/relu',
    #     'model/encoder/conv3/relu',
    #     'model/encoder/conv3bis/relu',
    #     'model/encoder/conv4/relu',
    #     'model/encoder/conv4bis/relu',
    #     'model/encoder/conv4bisbis/relu',
    #     'model/dnn/dnn1/leaky',
    #     'model/dnn/dnn2/leaky',
    #     'model/dnn/dnn3/leaky',
    #     'model/decoder/deconv5/relu',
    #     'model/decoder/deconv5bis/relu',
    #     'model/decoder/deconv6/relu',
    #     'model/decoder/deconv6bis/relu',
    #     'model/decoder/deconv7bis/relu',
    #     'model/decoder/deconv7bis/relu',
    #     'model/decoder/deconv8/relu',
    #     'model/decoder/deconv8bis/relu',
    #     'model/decoder/logits/relu',
    # ]
    # U-Net
    conserve_nodes = [
        'model/contractor/conv1/sigmoid',
        'model/contractor/conv1bis/sigmoid',
        'model/contractor/conv2/sigmoid',
        'model/contractor/conv2bis/sigmoid',
        'model/contractor/conv3/sigmoid',
        'model/contractor/conv3bis/sigmoid',
        'model/contractor/conv4/sigmoid',
        'model/contractor/conv4bis/sigmoid',
        'model/bottom/bot5/sigmoid',
        'model/bottom/bot5bis/sigmoid',
        'model/bottom/deconv1/sigmoid',
        'model/decontractor/conv6/sigmoid',
        'model/decontractor/conv6bis/sigmoid',
        'model/decontractor/deconv2/sigmoid',
        'model/decontractor/conv7/sigmoid',
        'model/decontractor/conv7bis/sigmoid',
        'model/decontractor/deconv3/sigmoid',
        'model/decontractor/conv8/sigmoid',
        'model/decontractor/conv8bis/sigmoid',
        'model/decontractor/deconv4/sigmoid',
        'model/decontractor/conv9/sigmoid',
        'model/decontractor/conv9bis/sigmoid',
        'model/decontractor/logits/relu',
    ]
    graph_def_dir = '/media/tomoserver/ZELIANG/20191015/'
    step = 29447
    paths = {
        'step': step,
        'perplexity': 10,  #default 30 usual range 5-50
        'niter': 5000,  #default 5000
        'working_dir': graph_def_dir,
        'ckpt_dir': graph_def_dir + 'ckpt/',
        'ckpt_path': graph_def_dir + 'ckpt/step{}'.format(step),
        'save_pb_dir': graph_def_dir + 'pb/',
        'save_pb_path': graph_def_dir + 'pb/step{}.pb'.format(step),
        'data_dir': './dummy/80/',
        'rlt_dir':  graph_def_dir + 'rlt/',
        'tsne_dir':  graph_def_dir + 'tsne/',
        'tsne_path':  graph_def_dir + 'tsne/',
    }

    partialRlt_and_diff(paths=paths, conserve_nodes=conserve_nodes)
    # tsne_partialRes_weights(params=paths, conserve_nodes=conserve_nodes, mode='2D')
    # tsne_partialRes_weights(params=paths, conserve_nodes=conserve_nodes, mode='3D')
    # weights_hists_2excel(path=ckpt_dir)
