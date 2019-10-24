import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
from util import check_N_mkdir
from itertools import product
from PIL import Image
from proc import _stride
from tqdm import tqdm


def reconstruct(stack, image_size=None, stride=None):
    """
    inputs:
    -------
        stack: (np.ndarray) stack of patches to reconstruct
        image_size: (tuple | list) height and width for the final reconstructed image
        stride: (int) herein should be the SAME stride step that one used for preprocess
    return:
    -------
        img: (np.ndarray) final reconstructed image
        nb_patches: (int) number of patches need to provide to this function
    """
    i_h, i_w = image_size[:2]  #e.g. (a, b)
    p_h, p_w = stack.shape[1:3]  #e.g. (x, h, w, 1)
    img = np.zeros((i_h, i_w))

    # compute the dimensions of the patches array
    n_h = (i_h - p_h) // stride + 1
    n_w = (i_w - p_w) // stride + 1

    for p, (i, j) in zip(stack, product(range(n_h), range(n_w))):
        img[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            img[i, j] /= float(min(i + stride, p_h, i_h - i) *
                               min(j + stride, p_w, i_w - j))
    return img


def freeze_ckpt_for_inference(paths=None, hyper=None, conserve_nodes=None):
    assert isinstance(paths, dict), 'The paths parameter expected a dictionnay but other type is provided'
    assert isinstance(hyper, dict), 'The hyper parameter expected a dictionnay but other type is provided'
    # clean graph first
    tf.reset_default_graph()
    # freeze ckpt then convert to pb
    input_ph = tf.placeholder(tf.float32, shape=[None, hyper['patch_size'], hyper['patch_size'], 1], name='input_ph')
    dropout_ph = tf.placeholder(tf.float32, shape=[None], name='dropout_ph')
    restorer = tf.train.import_meta_graph(
        paths['ckpt_path'] + '.meta',
        input_map={
            'input_pipeline/input_cond/Merge_1': input_ph,
            'dropout_prob': dropout_ph
        },
        clear_devices=True,
    )

    input_graph_def = tf.get_default_graph().as_graph_def()
    check_N_mkdir(paths['save_pb_dir'])
    check_N_mkdir(paths['optimized_pb_dir'])

    # freeze to pb
    with tf.Session() as sess:
        # restore variables
        restorer.restore(sess, paths['ckpt_path'])
        # convert variable to constant
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=conserve_nodes,
        )

        # save to pb
        tf.summary.FileWriter(paths['working_dir'] + 'tb/after_freeze', sess.graph)
        with tf.gfile.GFile(paths['save_pb_path'], 'wb') as f:  # 'wb' stands for write binary
            f.write(output_graph_def.SerializeToString())


def optimize_curve_for_inference(paths=None, conserve_nodes=None):
    assert isinstance(paths, dict), 'The paths parameter expected a dictionnay but other type is provided'
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.summary.FileWriter(paths['working_dir'] + 'tb/after_reset', sess.graph)
    check_N_mkdir(paths['optimized_pb_dir'])
    os.system(
    "python -m tensorflow.python.tools.optimize_for_inference --input {} --output {} --input_names='input_ph,dropout_ph' --output_names={}".format(
            paths['save_pb_path'], paths['optimized_pb_path'], conserve_nodes[0]))


def inference(inputs=None, conserve_nodes=None, paths=None, hyper=None):
    assert isinstance(paths, dict), 'The paths parameter expected a dictionnay but other type is provided'
    assert isinstance(hyper, dict), 'The hyper parameter expected a dictionnay but other type is provided'
    # CPU/GPU
    config_params = {}
    if hyper['device_option'] == 'cpu':
        config_params['config'] = tf.ConfigProto(device_count={'GPU': 0})
    elif 'specific' in hyper['device_option']:
        print('using GPU:{}'.format(hyper['device_option'].split(':')[-1]))
        config_params['config'] = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=hyper['device_option'].split(':')[-1]),
                                                 allow_soft_placement=True,
                                                 log_device_placement=False,
                                                 )

    tf.reset_default_graph()
    with tf.gfile.GFile(paths['optimized_pb_path'], 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())
    G = tf.Graph()
    output = np.empty((hyper['nb_patch'], hyper['patch_size'], hyper['patch_size']))

    with tf.Session(graph=G, **config_params) as sess:
        _ = tf.import_graph_def(graph_def_optimized, return_elements=[conserve_nodes[0]])  # note: this line can really clean all input_pipeline/or input what only is necessary
        print('Operations in Optimized Graph:')
        print([op.name for op in G.get_operations()])
        X = G.get_tensor_by_name('import/' + 'input_ph:0')
        y = G.get_tensor_by_name('import/' + 'model/decoder/logits/relu:0')
        do = G.get_tensor_by_name('import/' + 'dropout_ph:0')
        tf.summary.FileWriter(paths['working_dir'] + 'tb/after_optimize', sess.graph)
        # note: 1.throw up OpenMP error on Mac.
        for i in range(hyper['nb_batch']):
            try:
                _out = sess.run(y, feed_dict={X: inputs[i * hyper['batch_size']: (i + 1) * hyper['batch_size']], do: 1})
                output[i * hyper['batch_size']: (i + 1) * hyper['batch_size']] = np.squeeze(_out)
            except Exception as e:
                print(e)
                _out = sess.run(y, feed_dict={X: inputs[i * hyper['batch_size']:], do: 1})
                output[i * hyper['batch_size']:] = np.squeeze(_out)

    return output


def inference_recursive(inputs=None, conserve_nodes=None, paths=None, hyper=None):
    assert isinstance(conserve_nodes, list), 'conserve nodes should be a list'
    assert isinstance(inputs, list), 'inputs is expected to be a list of images for heterogeneous image size!'
    assert isinstance(paths, dict), 'paths should be a dict'
    assert isinstance(hyper, dict), 'hyper should be a dict'
    check_N_mkdir(paths['out_dir'])
    freeze_ckpt_for_inference(paths=paths, hyper=hyper, conserve_nodes=conserve_nodes)  # there's still some residual nodes
    optimize_curve_for_inference(paths=paths, conserve_nodes=conserve_nodes)  # clean residual nodes: gradients, td.data.pipeline...

    # calculate nb of patch per img
    img_shape = [i_sz.shape for i_sz in inputs]
    n_h=[]
    n_w=[]
    for i_h, i_w in img_shape:
        n_h.append((i_h - hyper['patch_size']) // hyper['stride'] + 1)
        n_w.append((i_w - hyper['patch_size']) // hyper['stride'] + 1)

    # set device
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

    # load graph
    tf.reset_default_graph()
    with tf.gfile.GFile(paths['optimized_pb_path'], 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())
    G = tf.Graph()
    l_out = []

    with tf.Session(graph=G, **config_params) as sess:
        _ = tf.import_graph_def(graph_def_optimized, return_elements=[conserve_nodes[0]])
        X = G.get_tensor_by_name('import/' + 'input_ph:0')
        y = G.get_tensor_by_name('import/' + 'model/decoder/logits/relu:0')
        do = G.get_tensor_by_name('import/' + 'dropout_ph:0')
        # compute the dimensions of the patches array

        for i, _input in tqdm(enumerate(inputs), desc='image'):
            # define batsh size
            if n_h[i] >= n_w[i]:  #note: could throw Resource_exhaustedError if batch_size >> 500
                hyper['nb_batch'] = n_h[i]
                hyper['batch_size'] = n_w[i]
            else:
                hyper['nb_batch'] = n_w[i]
                hyper['batch_size'] = n_h[i]
            print('\nbatch size:{}, nb_batch:{}'.format(hyper['batch_size'], hyper['nb_batch']))
            hyper['nb_patch'] = n_h[i] * n_w[i]

            # define output shape
            output = np.empty((hyper['nb_patch'], hyper['patch_size'], hyper['patch_size']))

            # construct the patches
            _input = _stride(_input, stride=1, patch_size=hyper['patch_size'])
            _input = np.expand_dims(_input, axis=3)

            # inference
            for j in tqdm(range(hyper['nb_batch']), desc='batch'):
                try:
                    _out = sess.run(y, feed_dict={
                        X: _input[j * hyper['batch_size']: (j + 1) * hyper['batch_size']],
                        do: 1
                    })
                    output[j * hyper['batch_size']: (j + 1) * hyper['batch_size']] = np.squeeze(_out)
                except Exception as e:
                    print(e)
                    _out = sess.run(y, feed_dict={
                        X: _input[j * hyper['batch_size']:],
                        do: 1
                    })
                    output[j * hyper['batch_size']:] = np.squeeze(_out)

            # recon
            output = reconstruct(output, image_size=img_shape[i], stride=hyper['stride'])  # outputs.shape should be [x, :, :]

            # save
            check_N_mkdir(paths['out_dir'])
            output = np.squeeze(output)
            Image.fromarray(output).save(paths['out_dir'] + 'step{}_{}.tif'.format(paths['step'], i))
            l_out.append(output)
    return l_out


if __name__ == '__main__':
    c_nodes = [
            'model/decoder/logits/relu',
        ]
    graph_def_dir = './logs/2019_10_19_bs300_ps80_lr0.0001_cs5_nc80_do0.1_act_leaky_aug_True/hour22/'

    # segment raw img per raw img
    l_bs = [800, 700, 600, 500, 400, 300, 200, 100]
    l_time = []
    l_inf = []
    l_step = [13580]
    step = l_step[0]
    for bs in l_bs:
        paths = {
            'step': None,
            'working_dir': graph_def_dir,
            'ckpt_dir': graph_def_dir + 'ckpt/',
            'ckpt_path': graph_def_dir + 'ckpt/step{}'.format(step),
            'save_pb_dir': graph_def_dir + 'pb/',
            'save_pb_path': graph_def_dir + 'pb/frozen_step{}.pb'.format(step),
            'optimized_pb_dir': graph_def_dir + 'optimize/',
            'optimized_pb_path': graph_def_dir + 'optimize/optimized_{}.pb'.format(step),
            'tflite_pb_dir': graph_def_dir + 'tflite/',
            'in_dir': './result/in/',
            'out_dir': './result/out/',
            'rlt_dir': graph_def_dir + 'rlt/',
            'GPU': 0,
            'inference_dir': './result/',
        }

        hyperparams = {
            'patch_size': 80,
            'batch_size': None,
            'nb_batch': None,
            'nb_patch': None,
            'stride': 1,
            'device_option': 'cpu',
        }

        freeze_ckpt_for_inference(paths=paths, hyper=hyperparams, conserve_nodes=c_nodes)  #there's still some residual nodes
        optimize_curve_for_inference(paths=paths, conserve_nodes=c_nodes)  #clean residual nodes: gradients, td.data.pipeline...
        inputs = np.asarray(Image.open('./raw/1.tif'))

        # calculate nb of patch per img
        img_size = inputs.shape
        i_h, i_w = img_size

        # compute the dimensions of the patches array
        n_h = (i_h - hyperparams['patch_size']) // hyperparams['stride'] + 1
        n_w = (i_w - hyperparams['patch_size']) // hyperparams['stride'] + 1
        hyperparams['nb_patch'] = n_h * n_w
        hyperparams['batch_size'] = bs  #note: maximize here to improve the inference speed
        hyperparams['nb_batch'] = hyperparams['nb_patch'] // hyperparams['batch_size'] + 1  #note: arbitrary nb
        #
        # if n_h >= n_w:
        #     hyperparams['nb_batch'] = n_h
        #     hyperparams['batch_size'] = n_w
        # else:
        #     hyperparams['nb_batch'] = n_w
        #     hyperparams['batch_size'] = n_h


        # construct the patches
        inputs = _stride(inputs, stride=1, patch_size=hyperparams['patch_size'])
        inputs = np.expand_dims(inputs, axis=3)
        start_time = time.time()
        print('Starting time {}s'.format(start_time))
        outputs = inference(inputs=inputs, paths=paths, hyper=hyperparams)
        print('Intermediate time {}s'.format(time.time() - start_time))
        l_inf.append((time.time() - start_time))
        outputs = reconstruct(outputs, image_size=img_size, stride=hyperparams['stride'])  #outputs.shape should be [x, :, :]
        l_time.append((time.time() - start_time))
        print('Segment an image in {}s'.format((time.time() - start_time)))
        check_N_mkdir('./result/')
        outputs = np.squeeze(outputs)
        Image.fromarray(outputs).save('./result/step{}_1.tif'.format(step))
    pd.DataFrame({'batch_size': l_bs, 'infer time': l_inf, 'recon time': l_time}).to_csv('./result/bs_time.csv')






