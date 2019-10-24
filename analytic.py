import tensorflow as tf
import numpy as np
import pandas as pd
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


def change_hist_bins():
    pass


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
    for j, layer_name in enumerate(wn + dnn_wn):
        df_w[layer_name.split(':')[0].replace('/', '_')] = pd.DataFrame({'0': _ws[j].flatten()})
    for j, layer_name in enumerate(bn + dnn_bn):
        df_b[layer_name.split(':')[0].replace('/', '_')] = pd.DataFrame({'0': _bs[j].flatten()})

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
    wn, bn, ws, bs, dnn_wn, dnn_bn, dnn_ws, dnn_bs = get_all_trainable_variables(lnames[0])
    step = [0]
    w_avg = [0]
    b_avg = [0]
    w_std = [0]
    b_std = [0]
    for ckpt_path in lnames[1:]:
        step.append(int(ckpt_path.split('step')[1].split('.')[0]))
        # get ws values at stepX
        wn_, bn_, ws_, bs_, dnn_wn_, dnn_bn_, dnn_ws_, dnn_bs_ = get_all_trainable_variables(ckpt_path)
        # program euclidean distance
        dis_w = []
        dis_b = []
        for w, w_ in zip(ws + dnn_ws, ws_ + dnn_ws_):
            try:
                # for CNN
                w, w_ = w.reshape((w.shape[0], w.shape[1], w.shape[2] * w.shape[3])), \
                        w_.reshape((w_.shape[0], w_.shape[1], w_.shape[2] * w_.shape[3]))
                for _w, _w_ in zip(w, w_):
                    dis_w.append(np.sqrt(np.sum(_w - _w_) ** 2))
            except:
                # for DNN
                dis_w.append(np.sqrt(np.sum(w - w_) ** 2))

        for b, b_ in zip(bs + dnn_bs, bs_ + dnn_bs_):
            dis_b.append(np.sqrt(np.sum(b - b_) ** 2))

        dis_w = np.asarray(dis_w)
        dis_b = np.asarray(dis_b)
        w_avg.append(dis_w.mean())
        w_std.append(dis_w.std())
        b_avg.append(dis_b.mean())
        b_std.append(dis_b.std())

    # create df
    dfs = {
        'weight': pd.DataFrame({
            'step': step,
            'avg': w_avg,
            'std': w_std,
        }),
        'bias': pd.DataFrame({
            'step': step,
            'avg': b_avg,
            'std': b_std,
        })

    }

    # write into excel
    with pd.ExcelWriter(rlt_dir + 'euclidean_dist.xlsx', engine='xlsxwriter') as writer:
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
        batch_size = 300

    new_ph = tf.placeholder(tf.float32, shape=[batch_size, patch_size, patch_size, 1], name='new_ph')

    # define nodes to conserve

    # convert ckpt to pb
    convert_ckpt2pb(input=new_ph, paths=paths, conserve_nodes=conserve_nodes)

    # load main graph
    g_main, ops_dict = load_mainGraph(conserve_nodes, path=paths['save_pb_path'])

    # run nodes and save results
    inference_and_save_partial_res(g_main, ops_dict, conserve_nodes, input_dir=paths['data_dir'], rlt_dir=paths['rlt_dir'] + 'step{}/'.format(paths['step']))


if __name__ == '__main__':
    # Xlearn
    conserve_nodes = [
        'model/encoder/conv1/relu',
        'model/encoder/conv1bis/relu',
        'model/encoder/conv2/relu',
        'model/encoder/conv2bis/relu',
        'model/encoder/conv3/relu',
        'model/encoder/conv3bis/relu',
        'model/encoder/conv4/relu',
        'model/encoder/conv4bis/relu',
        'model/encoder/conv4bisbis/relu',
        'model/dnn/dnn1/leaky',
        'model/dnn/dnn2/leaky',
        'model/dnn/dnn3/leaky',
        'model/decoder/deconv5/relu',
        'model/decoder/deconv5bis/relu',
        'model/decoder/deconv6/relu',
        'model/decoder/deconv6bis/relu',
        'model/decoder/deconv7bis/relu',
        'model/decoder/deconv7bis/relu',
        'model/decoder/deconv8/relu',
        'model/decoder/deconv8bis/relu',
        'model/decoder/logits/relu',
    ]
    # U-Net
    # conserve_nodes = [
    #     'model/contractor/conv1/sigmoid',
    #     'model/contractor/conv1bis/sigmoid',
    #     'model/contractor/conv2/sigmoid',
    #     'model/contractor/conv2bis/sigmoid',
    #     'model/contractor/conv3/sigmoid',
    #     'model/contractor/conv3bis/sigmoid',
    #     'model/contractor/conv4/sigmoid',
    #     'model/contractor/conv4bis/sigmoid',
    #     'model/bottom/bot5/sigmoid',
    #     'model/bottom/bot5bis/sigmoid',
    #     'model/bottom/deconv1/sigmoid',
    #     'model/decontractor/conv6/sigmoid',
    #     'model/decontractor/conv6bis/sigmoid',
    #     'model/decontractor/deconv2/sigmoid',
    #     'model/decontractor/conv7/sigmoid',
    #     'model/decontractor/conv7bis/sigmoid',
    #     'model/decontractor/deconv3/sigmoid',
    #     'model/decontractor/conv8/sigmoid',
    #     'model/decontractor/conv8bis/sigmoid',
    #     'model/decontractor/deconv4/sigmoid',
    #     'model/decontractor/conv9/sigmoid',
    #     'model/decontractor/conv9bis/sigmoid',
    #     'model/decontractor/logits/relu',
    # ]
    graph_def_dir = './logs/2019_10_19_bs300_ps80_lr0.0001_cs5_nc80_do0.1_act_leaky_aug_True/hour22/'
    step = 14580
    paths = {
        'step': step,
        'perplexity': 10,  #default 30 usual range 5-50
        'niter': 5000,  #default 5000
        'working_dir': graph_def_dir,
        'ckpt_dir': graph_def_dir + 'ckpt/',
        'ckpt_path': graph_def_dir + 'ckpt/step{}'.format(step),
        'save_pb_dir': graph_def_dir + 'pb/',
        'save_pb_path': graph_def_dir + 'pb/step{}.pb'.format(step),
        'data_dir': './proc/test/80/',
        'rlt_dir':  graph_def_dir + 'rlt/',
        'tsne_dir':  graph_def_dir + 'tsne/',
        'tsne_path':  graph_def_dir + 'tsne/',
    }

    partialRlt_and_diff(paths=paths, conserve_nodes=conserve_nodes)
    # tsne_partialRes_weights(params=paths, conserve_nodes=conserve_nodes, mode='2D')
    tsne_partialRes_weights(params=paths, conserve_nodes=conserve_nodes, mode='3D')
    # weights_hists_2excel(ckpt_dir=paths['ckpt_dir'], rlt_dir=paths['rlt_dir'])
    # weights_euclidean_distance(ckpt_dir=paths['ckpt_dir'], rlt_dir=paths['rlt_dir'])
