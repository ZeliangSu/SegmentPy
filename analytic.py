import pandas as pd
import numpy as np
import tensorflow as tf
import os
from util import get_all_trainable_variables, check_N_mkdir, print_nodes_name_shape, clean, plot_input_logit_label_diff, list_ckpts
from tsne import tsne, compare_tsne_2D, compare_tsne_3D
from inference import freeze_ckpt_for_inference
from PIL import Image
from scipy import interpolate
from writer import _resultWriter
from input import _one_hot, _minmaxscalar, _inverse_one_hot
from layers import customized_softmax_np

import logging
import log
logger = log.setup_custom_logger('root')
logger.setLevel(logging.WARNING)

# Xlearn
Xlearn_conserve_nodes = [
    'model/encoder/conv1/relu',
    'model/encoder/conv1bis/relu',
    'model/encoder/conv2/relu',
    'model/encoder/conv2bis/relu',
    'model/encoder/conv3/relu',
    'model/encoder/conv3bis/relu',
    'model/encoder/conv4/relu',
    'model/encoder/conv4bis/relu',
    'model/encoder/conv4bisbis/relu',
    'model/dnn/dnn1/relu',
    'model/dnn/dnn2/relu',
    'model/dnn/dnn3/relu',
    'model/decoder/deconv5/relu',
    'model/decoder/deconv5bis/relu',
    'model/decoder/deconv6/relu',
    'model/decoder/deconv6bis/relu',
    'model/decoder/deconv7bis/relu',
    'model/decoder/deconv7bis/relu',
    'model/decoder/deconv8/relu',
    'model/decoder/deconv8bis/relu',
    'model/decoder/logits/add',
]

# U-Net
Unet_conserve_nodes = [
    'model/contractor/conv1/leaky',
    'model/contractor/conv1bis/leaky',
    'model/contractor/conv2/leaky',
    'model/contractor/conv2bis/leaky',
    'model/contractor/conv3/leaky',
    'model/contractor/conv3bis/leaky',
    'model/contractor/conv4/leaky',
    'model/contractor/conv4bis/leaky',
    'model/bottom/bot5/leaky',
    'model/bottom/bot5bis/leaky',
    'model/bottom/deconv1/leaky',
    'model/decontractor/conv6/leaky',
    'model/decontractor/conv6bis/leaky',
    'model/decontractor/deconv2/leaky',
    'model/decontractor/conv7/leaky',
    'model/decontractor/conv7bis/leaky',
    'model/decontractor/deconv3/leaky',
    'model/decontractor/conv8/leaky',
    'model/decontractor/conv8bis/leaky',
    'model/decontractor/deconv4/leaky',
    'model/decontractor/conv9/leaky',
    'model/decontractor/conv9bis/leaky',
    'model/decontractor/logits/add',
]

# LRCS
LRCS_conserve_nodes = [
    'LRCS/encoder/conv1/leaky',
    'LRCS/encoder/conv1bis/leaky',
    'LRCS/encoder/conv2/leaky',
    'LRCS/encoder/conv2bis/leaky',
    'LRCS/encoder/conv3/leaky',
    'LRCS/encoder/conv3bis/leaky',
    'LRCS/encoder/conv4/leaky',
    'LRCS/encoder/conv4bis/leaky',
    'LRCS/encoder/conv4bisbis/leaky',
    'LRCS/dnn/dnn1/leaky',
    'LRCS/dnn/dnn2/leaky',
    'LRCS/dnn/dnn3/leaky',
    'LRCS/decoder/deconv5/leaky',
    'LRCS/decoder/deconv5bis/leaky',
    'LRCS/decoder/deconv6/leaky',
    'LRCS/decoder/deconv6bis/leaky',
    'LRCS/decoder/deconv7/leaky',
    'LRCS/decoder/deconv7bis/leaky',
    'LRCS/decoder/deconv8/leaky',
    'LRCS/decoder/deconv8bis/leaky',
    'LRCS/decoder/logits/identity',
]

conserve_nodes_dict = {
    'Xlearn': Xlearn_conserve_nodes,
    'Unet': Unet_conserve_nodes,
    'LRCS': LRCS_conserve_nodes
}


def load_mainGraph(conserve_nodes, path='./dummy/pb/test.pb'):
    """
    inputs:
    -------
        conserve_nodes: (list of string)
        path: (str)

    return:
    -------
        g_main: (tf.Graph())
        ops_dict: (dictionary of operations)
    """
    # import graph def
    with tf.gfile.GFile(path, mode='rb') as f:
        # init GraphDef()
        restored_graph_def = tf.GraphDef()
        # parse saved .pb to GraphDef()
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as g_main:
        # import graph def
        tf.import_graph_def(
            graph_def=restored_graph_def,
            return_elements=[conserve_nodes[-1]],
            name=''  # note: '' so that won't have import/ prefix
        )

    # prepare feed_dict for inference
    ops_dict = {
        'ops': [g_main.get_tensor_by_name(op_name + ':0') for op_name in conserve_nodes],
        }
    return g_main, ops_dict


def inference_and_save_partial_res(g_main, ops_dict, conserve_nodes, hyper=None, input_dir=None, rlt_dir=None):
    """

    Parameters
    ----------
    g_combined: (tf.Graph())
    ops_dict: (list of operations)
    conserve_nodes: (list of string)

    Returns
    -------
        None

    """
    config_params = {}
    if hyper['device_option'] == 'cpu':
        config_params['config'] = tf.ConfigProto(device_count={'GPU': 0})
    elif 'specific' in hyper['device_option']:
        print('using GPU:{}'.format(hyper['device_option'].split(':')[-1]))
        config_params['config'] = tf.ConfigProto(
            gpu_options=tf.GPUOptions(visible_device_list=hyper['device_option'].split(':')[-1]),
            allow_soft_placement=True,
            log_device_placement=False,
            )
    with g_main.as_default() as g_main:
        # init a writer class
        plt_illd = plot_input_logit_label_diff()

        new_input = g_main.get_tensor_by_name('new_input:0')

        # write firstly input and output images
        imgs = [
            np.asarray(Image.open(input_dir + '0.tif'))[i * 100:512 + i*100, i*100:512 + i*100] for i in range(hyper['batch_size'])  #fixme: better use test dataset
        ]
        plt_illd.add_input(np.asarray(imgs))
        _resultWriter(imgs, 'input', path=rlt_dir)

        labels = [np.asarray(Image.open(input_dir + '0_label.tif'))[i * 100:512 + i*100, i*100:512 + i*100] for i in range(hyper['batch_size'])]   #fixme: better use test dataset
        plt_illd.add_label(np.asarray(labels))
        _resultWriter(labels, 'label', path=rlt_dir)

        img_size = hyper['patch_size']

        # prepare feed_dict
        feed_dict = {
            new_input: np.array(imgs).reshape((hyper['batch_size'], img_size, img_size, 1)),
        }
        if hyperparams['batch_normalization']:
            new_BN_phase = g_main.get_tensor_by_name('new_BN:0')
            feed_dict[new_BN_phase] = False

        try:
            dropout_input = g_main.get_tensor_by_name('new_dropout:0')
            feed_dict[dropout_input] = 1.0

        except Exception as e:
            print('Error(message):', e)
            pass

        # run inference
        with tf.Session(graph=g_main, **config_params) as sess:
            print_nodes_name_shape(sess.graph)
            # run partial results operations and diff block
            res = sess.run(ops_dict['ops'], feed_dict=feed_dict)
            activations = []

            # note: save partial/final inferences of the first image
            for layer_name, tensors in zip(conserve_nodes, res):
                try:
                    if tensors.ndim == 4 or 2:
                        if 'logit' in layer_name:
                            tensors = customized_softmax_np(tensors)
                            tensors = _inverse_one_hot(tensors)
                            plt_illd.add_logit(tensors)

                        else:
                            tensors = [np.squeeze(tensors[i]) for i in range(tensors.shape[0])]
                except Exception as e:
                    print(e)
                    pass
                _resultWriter(tensors, layer_name=layer_name.split('/')[-2],
                              path=rlt_dir)  # for cnn outputs shape: [batch, w, h, nb_conv]
                activations.append(tensors)

    # calculate diff by numpy
    # res[-1] final result
    if hyper['mode'] == 'regression':
        res_diff = np.equal(np.asarray(np.squeeze(res[-1]), dtype=np.int), np.asarray(labels))
        res_diff = np.asarray(res_diff, dtype=np.int)
        # note: save diff of all imgs
        plt_illd.add_diff(np.asarray(res_diff))
        _resultWriter(np.transpose(res_diff, (1, 2, 0)), 'diff',
                      path=rlt_dir)  # for diff output shape: [batch, w, h, 1]
    else:
        # one-hot the label
        labels = np.expand_dims(np.asarray(labels), axis=3)  # list --> array --> (B, H, W, 1)
        logits = customized_softmax_np(np.asarray(res[-1], dtype=np.int))  # (B, H, W, 3)
        res_diff = np.equal(_inverse_one_hot(clean(logits)), labels)  #(B, H, W)
        plt_illd.add_diff(res_diff.astype(int))
        _resultWriter(res_diff.astype(int), 'diff', path=rlt_dir)  # for diff output shape: [batch, w, h, 3]

    check_N_mkdir(rlt_dir + 'illd/')
    plt_illd.plot(out_path=rlt_dir + 'illd/illd.tif')
    # return
    return activations


def visualize_weights(params=None, plt=False, mode='copy'):
    assert isinstance(params, dict)
    dir = params['rlt_dir'] + 'weights/step{}/'.format(params['step'])
    wn, _, ws, _, _, _, _, _ = get_all_trainable_variables(params['ckpt_path'])
    for _wn, _w in zip(wn, ws):
        for i in range(_w.shape[3]):
            if mode == 'interpolation':
                # interpolation and enlarge to a bigger matrix (instead of repeating)
                x = np.linspace(-1, 1, _w.shape[0])
                y = np.linspace(-1, 1, _w.shape[1])
                f = interpolate.interp2d(x, y, np.sum(_w[:, :, :, i], axis=2), kind='cubic')
                x = np.linspace(-1, 1, _w.shape[0] * 30)
                y = np.linspace(-1, 1, _w.shape[1] * 30)
                tmp = f(x, y)

            elif mode == 'copy':
                tmp = np.repeat(np.repeat(np.sum(_w[:, :, :, i], axis=2), 30, axis=0), 30, axis=1)

            else:
                raise NotImplementedError('mode??')
            # save
            check_N_mkdir(dir + '{}/'.format(_wn.split('/')[0]))
            Image.fromarray(tmp).save(
                dir + '{}/{}.tif'.format(_wn.split('/')[0], i))
        if plt:
            pass


def tsne_on_bias(params=None, mode='2D'):
    assert params != None, 'please define the dictionary of paths'
    assert isinstance(params, dict), 'paths should be a dictionary containning path'

    # get bias
    _, bn_init, _, bs_init, _, dnn_bn_init, _, dnn_bs_init = get_all_trainable_variables(params['ckpt_path_init'])
    _, bn, _, bs, _, dnn_bn, _, dnn_bs = get_all_trainable_variables(params['ckpt_path'])

    shapes = [b.shape[0] for b in bs + dnn_bs]
    max_shape = 0
    for _shape in shapes:
        if _shape >= max_shape:
            max_shape = _shape

    new_bn = []
    new_bs = []
    grps = []
    which = []

    # preparation: unify the b shape by padding
    # for first ckpt
    for _bn, _b in zip(bn_init + dnn_bn_init, bs_init + dnn_bs_init):
        new_bn.append(_bn.split(':')[0])
        grps.append(_bn.split('/')[0])
        which.append(0)

        # pad
        if _b.shape[0] < max_shape:
            _b = np.pad(_b, (0, max_shape - _b.shape[0]), constant_values=0)
        new_bs.append(_b)

    # for second ckpt
    for _bn, _b in zip(bn + dnn_bn, bs + dnn_bs):
        new_bn.append(_bn.split(':')[0])
        grps.append(_bn.split('/')[0])
        which.append(1)

        # pad
        if _b.shape[0] < max_shape:
            _b = np.pad(_b, (0, max_shape - _b.shape[0]), constant_values=0)
        new_bs.append(_b)

    # inject into t-SNE
    res = tsne(
        np.asarray(new_bs).reshape(len(new_bs), -1),
        perplexity=params['perplexity'],
        niter=params['niter'],
        mode=mode,
    )

    # mkdir
    check_N_mkdir(params['rlt_dir'])

    # visualize the tsne
    if mode == '2D':
        compare_tsne_2D(res, new_bn, grps, which=which, rlt_dir=params['tsne_dir'], preffix='Bias', fst=paths['ckpt_path_init'].split('step')[1], sec=paths['ckpt_path'].split('step')[1])
    elif mode == '3D':
        compare_tsne_3D(res, new_bn, grps, which=which, rlt_dir=params['tsne_dir'], suffix=params['step'])
    else:
        raise NotImplementedError('please choose 2D or 3D mode')


def tsne_on_weights(params=None, mode='2D'):
    """
    input:
    -------
        ckptpath: (string) path to the checkpoint that we convert to .pb. e.g. './logs/YYYY_MM_DD_.../hourHH/ckpt/step{}'
    return:
    -------
        None
    """
    assert params!=None, 'please define the dictionary of paths'
    assert isinstance(params, dict), 'paths should be a dictionary containning path'
    # run tsne on wieghts
    # get weights from checkpoint
    wns_init, _, ws_init, _, _, _, _, _ = get_all_trainable_variables(params['ckpt_path_init'])
    wns, _, ws, _, _, _, _, _ = get_all_trainable_variables(params['ckpt_path'])

    # arange label and kernel
    new_wn = []
    new_ws = []
    grps = []
    which = []

    # for 1st ckpt
    for wn, w in zip(wns_init, ws_init):  # w.shape = [c_w, c_h, c_in, nb_conv]
        for i in range(w.shape[3]):
            new_wn.append(wn + '_{}'.format(i))  # e.g. conv4bis_96
            grps.append(wn.split('/')[0])
            which.append(0)

            #note: associativity: a x b + a x c = a x (b + c)
            # "...a kernel is the sum of all the dimensions in the previous layer..."
            # https://stackoverflow.com/questions/42712219/dimensions-in-convolutional-neural-network
            new_ws.append(np.sum(w[:, :, :, i], axis=2))  # e.g. (3, 3, 12, 24) [w, h, in, nb_conv] --> (3, 3, 24)

    # for 2nd ckpt
    for wn, w in zip(wns, ws):  # w.shape = [c_w, c_h, c_in, nb_conv]
        for i in range(w.shape[3]):
            new_wn.append(wn + '_{}'.format(i))  # e.g. conv4bis_96
            grps.append(wn.split('/')[0])
            which.append(1)
            new_ws.append(np.sum(w[:, :, :, i], axis=2))  # e.g. (3, 3, 12, 24) [w, h, in, nb_conv] --> (3, 3, 24)

    # inject into t-SNE
    res = tsne(np.array(new_ws).transpose((1, 2, 0)).reshape(len(new_ws), -1),
               perplexity=params['perplexity'],
               niter=params['niter'], mode=mode)  # e.g. (3, 3, x) --> (9, x) --> (x, 2) or (x, 3)

    # mkdir
    check_N_mkdir(params['rlt_dir'])
    # visualize the tsne
    if mode == '2D':
        compare_tsne_2D(res, new_wn, grps, which, rlt_dir=params['tsne_dir'], fst=paths['ckpt_path_init'].split('step')[1], sec=paths['ckpt_path'].split('step')[1])
    elif mode == '3D':
        compare_tsne_3D(res, new_wn, grps, which, rlt_dir=params['tsne_dir'], suffix=params['step'])
    else:
        raise NotImplementedError('please choose 2D or 3D mode')


def weights_hists_2excel(ckpt_dir=None, rlt_dir=None):
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
    check_N_mkdir(rlt_dir)
    lnames = []
    for step in os.listdir(ckpt_dir):
        if step.endswith('.meta'):
            lnames.append(ckpt_dir + step.split('.')[0])
    assert len(lnames) > 1, 'The ckpt directory should have at least 2 ckpts!'
    lnames = sorted(lnames)
    # fixme: ValueError: This sheet is too large! Your sheet size is: 1280000, 1 Max sheet size is: 1048576, 16384

    bins = 1000
    step = []
    _min = {}  # [conv1w, conv1b...]
    _max = {}  # [conv1w, conv1b...]
    df_w = {}  # {conv1_w: pd.DataFrame({0:..., 1000:...}), conv1bis_w: pd.DataFrame({0:..., 1000:..., ...})}
    df_b = {}  # {conv1_b: pd.DataFrame({0:..., 1000:...}), conv1bis_b: pd.DataFrame({0:..., 1000:..., ...})}
    hist_w = {}  # {conv1_w: pd.DataFrame({x:..., 0:..., 1000:...}), conv1bis_w: pd.DataFrame({x:..., 0:..., 1000:..., ...})}
    hist_b = {}  # {conv1_b: pd.DataFrame({x:..., 0:..., 1000:...}), conv1bis_b: pd.DataFrame({x:..., 0:..., 1000:..., ...})}

    # step 0
    wn, bn, ws, bs, dnn_wn, dnn_bn, dnn_ws, dnn_bs = get_all_trainable_variables(lnames[0])
    _ws = ws + dnn_ws
    _bs = bs + dnn_bs
    step.append(lnames[0].split('step')[1].split('.')[0])

    # init dataframes
    for i, layer_name in enumerate(wn + dnn_wn):
        df_w[layer_name.split(':')[0].replace('/', '_')] = pd.DataFrame({'0': _ws[i].flatten()})
    for i, layer_name in enumerate(bn + dnn_bn):
        df_b[layer_name.split(':')[0].replace('/', '_')] = pd.DataFrame({'0': _bs[i].flatten()})

    # add more step to layers params
    for i, ckpt_path in enumerate(lnames[1:]):
        step.append(ckpt_path.split('step')[1].split('.')[0])

        # get weights-bias names and values
        wn, bn, ws, bs, dnn_wn, dnn_bn, dnn_ws, dnn_bs = get_all_trainable_variables(ckpt_path)
        _ws = ws + dnn_ws
        _bs = bs + dnn_bs

        # insert values
        for j, layer_name in enumerate(wn + dnn_wn):
            df_w[layer_name.split(':')[0].replace('/', '_')].insert(i + 1, step[i + 1], _ws[j].flatten())
        for j, layer_name in enumerate(bn + dnn_bn):
            df_b[layer_name.split(':')[0].replace('/', '_')].insert(i + 1, step[i + 1], _bs[j].flatten())

    # calculate histogram
    # find min and max of w/b of each layer
    for j, layer_name in enumerate(wn + dnn_wn):
        _min[layer_name.split(':')[0].replace('/', '_')] = df_w[layer_name.split(':')[0].replace('/', '_')].min()
        _max[layer_name.split(':')[0].replace('/', '_')] = df_w[layer_name.split(':')[0].replace('/', '_')].max()

    for j, layer_name in enumerate(bn + dnn_bn):
        _min[layer_name.split(':')[0].replace('/', '_')] = df_b[layer_name.split(':')[0].replace('/', '_')].min()
        _max[layer_name.split(':')[0].replace('/', '_')] = df_b[layer_name.split(':')[0].replace('/', '_')].max()

    # get histogram of W
    for layer_name in wn + dnn_wn:
        _, _edge = np.histogram(
            np.asarray(df_w[layer_name.split(':')[0].replace('/', '_')]),
            bins=np.linspace(
                _min[layer_name.split(':')[0].replace('/', '_')][0],
                _max[layer_name.split(':')[0].replace('/', '_')][0],
                bins
            )
        )
        hist_w[layer_name.split(':')[0].replace('/', '_')] = pd.DataFrame({'x': _edge[1:]})
        i = 0
        for _step, params in df_w[layer_name.split(':')[0].replace('/', '_')].iteritems():
            _hist, _ = np.histogram(
                np.asarray(params),
                bins=np.linspace(_min[layer_name.split(':')[0].replace('/', '_')][_step],
                                 _max[layer_name.split(':')[0].replace('/', '_')][_step],
                                 num=bins
                                 )
            )
            hist_w[layer_name.split(':')[0].replace('/', '_')].insert(i + 1, _step, _hist)
            i += 1
    # clean instance
    del df_w

    # get histogram of b
    for layer_name in bn + dnn_bn:
        _hist, _edge = np.histogram(
            np.asarray(df_b[layer_name.split(':')[0].replace('/', '_')]),
            bins=np.linspace(
                _min[layer_name.split(':')[0].replace('/', '_')][0],
                _max[layer_name.split(':')[0].replace('/', '_')][0],
                bins
            )
        )
        hist_b[layer_name.split(':')[0].replace('/', '_')] = pd.DataFrame({'x': _edge[1:]})
        i = 0
        for _step, params in df_b[layer_name.split(':')[0].replace('/', '_')].iteritems():
            _hist, _edge = np.histogram(
                np.asarray(params),
                bins=np.linspace(
                    _min[layer_name.split(':')[0].replace('/', '_')][_step],
                    _max[layer_name.split(':')[0].replace('/', '_')][_step],
                    bins)
            )
            hist_b[layer_name.split(':')[0].replace('/', '_')].insert(i + 1, _step, _hist)
            i += 1
    # clean instance
    del df_b

    # write into excel
    check_N_mkdir(rlt_dir + 'weight_hist/')
    for xlsx_name in hist_w.keys():
        with pd.ExcelWriter(rlt_dir + 'weight_hist/{}.xlsx'.format(xlsx_name), engine='xlsxwriter') as writer:
            hist_w[xlsx_name].to_excel(writer, index=False)
    for xlsx_name in hist_b.keys():
        with pd.ExcelWriter(rlt_dir + 'weight_hist/{}.xlsx'.format(xlsx_name), engine='xlsxwriter') as writer:
            hist_b[xlsx_name].to_excel(writer, index=False)


def weights_euclidean_distance(ckpt_dir=None, rlt_dir=None):
    """
    inputs:
    -------
        path: (string) path to get the checkpoint e.g. './logs/YYYY_MM_DD_.../hourHH/ckpt/'
    return:
    -------
        None
    """
    # construct dataframe
    # header sheet_name weight: [step0, step20, ...]
    # header sheet_name bias: [step0, step20, ...]
    check_N_mkdir(rlt_dir)
    lnames = []
    for step in os.listdir(ckpt_dir):
        if step.endswith('.meta'):
            lnames.append(ckpt_dir + step.split('.')[0])
    lnames = sorted(lnames)

    # get weights-bias values at step0
    wn, bn, ws_init, bs_init, dnn_wn, dnn_bn, dnn_ws_init, dnn_bs_init = get_all_trainable_variables(lnames[0])
    print('\n ********* processing euclidean distance for each checkpoint')
    l_total_w_avg = [0]
    l_total_b_avg = [0]
    l_total_w_std = [0]
    l_total_b_std = [0]

    dic_w = {'step': [0]}
    dic_b = {'step': [0]}

    for key in wn + dnn_wn:
        dic_w[key.split('/')[0] + '_avg'] = [0]
        dic_w[key.split('/')[0] + '_std'] = [0]
        dic_b[key.split('/')[0] + '_avg'] = [0]
        dic_b[key.split('/')[0] + '_std'] = [0]

    for ckpt_path in lnames[1:]:
        # insert step
        step = int(ckpt_path.split('step')[1].split('.')[0])
        print(step)
        dic_w['step'].append(step)
        dic_b['step'].append(step)
        total_dis_w = []
        total_dis_b = []

        # get ws values at stepX
        wn, bn, ws_, bs_, dnn_wn, dnn_bn, dnn_ws_, dnn_bs_ = get_all_trainable_variables(ckpt_path)
        # program euclidean distance
        # for w
        for _wn, w_init, w_ in zip(wn + dnn_wn, ws_init + dnn_ws_init, ws_ + dnn_ws_):
            l_dis_w = []
            try:
                # for CNN
                # retrive the filters
                w_init, w_ = np.sum(w_init, axis=2), np.sum(w_, axis=2)
                # write w
                for i in range(w_init.shape[2]):
                    dis_w = np.sqrt(np.sum((w_init[:, :, i] - w_[:, :, i]) ** 2))
                    l_dis_w.append(dis_w)
                    total_dis_w.append(dis_w)

            except Exception as e:
                # for DNN
                dis_w = np.sqrt(np.sum((w_init - w_) ** 2))
                l_dis_w.append(dis_w)
                total_dis_w.append(dis_w)

            # save w into dfs
            dic_w[_wn.split('/')[0] + '_avg'].append(np.asarray(l_dis_w).mean())
            dic_w[_wn.split('/')[0] + '_std'].append(np.asarray(l_dis_w).std())

        # for b
        for _bn, b_init, b_ in zip(bn + dnn_bn, bs_init + dnn_bs_init, bs_ + dnn_bs_):
            l_dis_b = []
            for i in range(b_init.shape[0]):
                dis_b = np.sqrt(np.sum((b_init[i] - b_[i]) ** 2))
                l_dis_b.append(dis_b)
                total_dis_b.append(dis_b)

            # write b into dfs
            dic_b[_bn.split('/')[0] + '_avg'].append(np.asarray(l_dis_b).mean())
            dic_b[_bn.split('/')[0] + '_std'].append(np.asarray(l_dis_b).std())
        l_total_w_avg.append(np.asarray(total_dis_w).mean())
        l_total_w_std.append(np.asarray(total_dis_w).std())
        l_total_b_avg.append(np.asarray(total_dis_b).mean())
        l_total_b_std.append(np.asarray(total_dis_b).std())

    dic_w['total_avg'] = l_total_w_avg
    dic_w['total_std'] = l_total_w_std
    dic_b['total_avg'] = l_total_b_avg
    dic_b['total_std'] = l_total_b_std

    # create df
    try:
        dfs = {'weight': pd.DataFrame(dic_w), 'bias': pd.DataFrame(dic_b)}
    except Exception as e:
        #note: in a BN network, there are less bias
        logger.info(e)
        dfs = {'weight': pd.DataFrame(dic_w)}

    # write into excel
    with pd.ExcelWriter(rlt_dir + 'euclidean_dist.xlsx', engine='xlsxwriter') as writer:
        for sheet_name in dfs.keys():
            dfs[sheet_name].sort_values('step').to_excel(writer, sheet_name=sheet_name, index=False)


def weights_angularity(ckpt_dir=None, rlt_dir=None):
    """
    inputs:
    -------
        path: (string) path to get the checkpoint e.g. './logs/YYYY_MM_DD_.../hourHH/ckpt/'
    return:
    -------
        None
    """
    # construct dataframe
    # header sheet_name weight: [step0, step20, ...]
    # header sheet_name bias: [step0, step20, ...]
    check_N_mkdir(rlt_dir)
    lnames = []
    for step in os.listdir(ckpt_dir):
        if step.endswith('.meta'):
            lnames.append(ckpt_dir + step.split('.')[0])
    lnames = sorted(lnames)

    # get weights-bias values at step0
    wn, bn, ws_init, bs_init, dnn_wn, dnn_bn, dnn_ws_init, dnn_bs_init = get_all_trainable_variables(lnames[0])
    print('\n ********* processing angularity for each checkpoint')
    l_total_w_avg = [0]
    l_total_b_avg = [0]
    l_total_w_std = [0]
    l_total_b_std = [0]

    dic_w = {'step': [0]}
    dic_b = {'step': [0]}

    for key in wn + dnn_wn:
        dic_w[key.split('/')[0] + '_avg'] = [0]
        dic_w[key.split('/')[0] + '_std'] = [0]
        dic_b[key.split('/')[0] + '_avg'] = [0]
        dic_b[key.split('/')[0] + '_std'] = [0]

    for ckpt_path in lnames[1:]:
        # insert step
        step = int(ckpt_path.split('step')[1].split('.')[0])
        print(step)
        dic_w['step'].append(step)
        dic_b['step'].append(step)
        total_ang_w = []
        total_ang_b = []

        # get ws values at stepX
        wn, bn, ws_, bs_, dnn_wn, dnn_bn, dnn_ws_, dnn_bs_ = get_all_trainable_variables(ckpt_path)
        # program cosine alpha

        # for w
        for _wn, w_init, w_ in zip(wn + dnn_wn, ws_init + dnn_ws_init, ws_ + dnn_ws_):
            l_ang_w = []
            try:
                # for CNN
                # retrive the filters
                w_init, w_ = np.sum(w_init, axis=2), np.sum(w_, axis=2)
                # write w
                for i in range(w_init.shape[2]):
                    # note: need to flatten the kernel
                    angle_w = np.dot(w_init[:, :, i].ravel(), w_[:, :, i].ravel()) / (np.linalg.norm(w_init[:, :, i].ravel()) * np.linalg.norm(w_[:, :, i].ravel()))
                    l_ang_w.append(angle_w)
                    total_ang_w.append(angle_w)

            except Exception as e:
                # for DNN
                # Retrieve weights
                w_init, w_ = np.sum(w_init,  axis=1), np.sum(w_, axis=1)
                angle_w = np.dot(w_init.T, w_) / (np.linalg.norm(w_init) * np.linalg.norm(w_))
                l_ang_w.append(angle_w)
                total_ang_w.append(angle_w)

            # save w into dfs
            dic_w[_wn.split('/')[0] + '_avg'].append(np.asarray(l_ang_w).mean())
            dic_w[_wn.split('/')[0] + '_std'].append(np.asarray(l_ang_w).std())

        # for b
        for _bn, b_init, b_ in zip(bn + dnn_bn, bs_init + dnn_bs_init, bs_ + dnn_bs_):
            l_ang_b = []

            ang_b = np.dot(b_init.ravel(), b_.ravel()) / (np.linalg.norm(b_init) * np.linalg.norm(b_))
            l_ang_b.append(ang_b)
            total_ang_b.append(ang_b)

            # write b into dfs
            dic_b[_bn.split('/')[0] + '_avg'].append(np.asarray(l_ang_b).mean())
            dic_b[_bn.split('/')[0] + '_std'].append(np.asarray(l_ang_b).std())
        l_total_w_avg.append(np.asarray(total_ang_w).mean())
        l_total_w_std.append(np.asarray(total_ang_w).std())
        l_total_b_avg.append(np.asarray(total_ang_b).mean())
        l_total_b_std.append(np.asarray(total_ang_b).std())

    dic_w['total_avg'] = l_total_w_avg
    dic_w['total_std'] = l_total_w_std
    dic_b['total_avg'] = l_total_b_avg
    dic_b['total_std'] = l_total_b_std

    # create df
    try:
        dfs = {'weight': pd.DataFrame(dic_w), 'bias': pd.DataFrame(dic_b)}
    except Exception as e:
        #note: in a BN network, there are less bias
        logger.info(e)
        dfs = {'weight': pd.DataFrame(dic_w)}

    # write into excel
    with pd.ExcelWriter(rlt_dir + 'angularity.xlsx', engine='xlsxwriter') as writer:
        for sheet_name in dfs.keys():
            dfs[sheet_name].sort_values('step').to_excel(writer, sheet_name=sheet_name, index=False)


def partialRlt_and_diff(paths=None, hyperparams=None, conserve_nodes=None, plt=False):
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

    # clean graph first
    tf.reset_default_graph()

    # convert ckpt to pb
    freeze_ckpt_for_inference(paths=paths, hyper=hyperparams, conserve_nodes=conserve_nodes)

    # load main graph
    g_main, ops_dict = load_mainGraph(conserve_nodes, path=paths['save_pb_path'])

    # run nodes and save results
    inference_and_save_partial_res(g_main, ops_dict, conserve_nodes,
                                   input_dir=paths['data_dir'],
                                   rlt_dir=paths['rlt_dir'] + 'p_inference/step{}/'.format(paths['step']),
                                   hyper=hyperparams)

    # plt
    if plt:
        # todo: plot top 10 activations
        pass


if __name__ == '__main__':
    # disable the GPU if there's a traning
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    hyperparams = {
        'patch_size': 512,
        'batch_size': 8,
        'nb_batch': None,
        'nb_patch': None,
        'stride': 1,
        'device_option': 'cpu',
        'mode': 'classification',
        'batch_normalization': False,
    }
    conserve_nodes = conserve_nodes_dict['LRCS']
    graph_def_dir = '/Users/suweiguo/Desktop/model_20200219_hour8/'
    step = 0
    step_init = 0
    paths = {
        'step': step,
        'perplexity': 100,  #default 30 usual range 5-50
        'niter': 5000,  #default 5000
        'working_dir': graph_def_dir,
        'ckpt_dir': graph_def_dir + 'ckpt/',
        'ckpt_path': graph_def_dir + 'ckpt/step{}'.format(step_init),
        'save_pb_dir': graph_def_dir + 'pb/',
        'save_pb_path': graph_def_dir + 'pb/step{}.pb'.format(step_init),
        'data_dir': './raw/', #todo:
        'rlt_dir':  graph_def_dir + 'rlt/',
        'tsne_dir':  graph_def_dir + 'tsne/',
        'tsne_path':  graph_def_dir + 'tsne/',
    }
    print('Proceed step {}'.format(paths['step']))
    # visualize_weights(params=paths)
    # partialRlt_and_diff(paths=paths, hyperparams=hyperparams, conserve_nodes=conserve_nodes)

    l_step = list_ckpts(graph_def_dir + 'ckpt/')
    for step in l_step:
        paths = {
            'step': step,
            'perplexity': 100,  #default 30 usual range 5-50
            'niter': 5000,  #default 5000
            'working_dir': graph_def_dir,
            'ckpt_dir': graph_def_dir + 'ckpt/',
            'ckpt_path': graph_def_dir + 'ckpt/step{}'.format(step),
            'ckpt_path_init': graph_def_dir + 'ckpt/step{}'.format(step_init),
            'save_pb_dir': graph_def_dir + 'pb/',
            'save_pb_path': graph_def_dir + 'pb/step{}.pb'.format(step),
            'data_dir': './raw/',
            'rlt_dir':  graph_def_dir + 'rlt/',
            'tsne_dir':  graph_def_dir + 'tsne/',
            'tsne_path':  graph_def_dir + 'tsne/',
        }
        print('Proceed step {}'.format(paths['step']))
        # visualize_weights(params=paths)
        partialRlt_and_diff(paths=paths, hyperparams=hyperparams, conserve_nodes=conserve_nodes)
        # tsne_on_weights(params=paths, mode='2D')
        # tsne_on_bias(params=paths, mode='2D')
        # weights_euclidean_distance(ckpt_dir=paths['ckpt_dir'], rlt_dir=paths['rlt_dir'])
        # weights_angularity(ckpt_dir=paths['ckpt_dir'], rlt_dir=paths['rlt_dir'])
        # weights_hists_2excel(ckpt_dir=paths['ckpt_dir'], rlt_dir=paths['rlt_dir'])
