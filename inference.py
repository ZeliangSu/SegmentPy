import tensorflow as tf
import numpy as np
import os
from util import check_N_mkdir
from tensorflow.core.framework import graph_pb2
from itertools import product
import copy


def reconstruct(stack, image_size, step):
    """
    inputs:
    -------
        stack: (np.ndarray) stack of patches to reconstruct
        image_size: (tuple | list) height and width for the final reconstructed image
        step: (int) herein should be the SAME stride step that one used for preprocess
    return:
    -------
        img: (np.ndarray) final reconstructed image
        nb_patches: (int) number of patches need to provide to this function
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = stack.shape[1:3]
    img = np.zeros(image_size)

    # compute the dimensions of the patches array
    n_h = (i_h - p_h) // step + 1
    n_w = (i_w - p_w) // step + 1
    nb_patches = n_h * n_w

    for p, (i, j) in zip(stack, product(range(n_h), range(n_w))):
        img[i * step:i * step + p_h, j * step:j * step + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            img[i, j] /= float(min(i + step, p_h, i_h - i) *
                               min(j + step, p_w, i_w - j))
    return img, nb_patches


def freeze_ckpt_for_inference(paths=None, hyper=None):
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


def optimize_curve_for_inference(paths=None):
    assert isinstance(paths, dict), 'The paths parameter expected a dictionnay but other type is provided'
    tf.reset_default_graph()
    with tf.Session() as sess:
        tf.summary.FileWriter(paths['working_dir'] + 'tb/after_reset', sess.graph)
    check_N_mkdir(paths['optimized_pb_dir'])
    os.system(
    "python -m tensorflow.python.tools.optimize_for_inference --input {} --output {} --input_names='input_ph,dropout_ph' --output_names={}".format(
            paths['save_pb_path'], paths['optimized_pb_path'], conserve_nodes[0]))


def inference(inputs=None, paths=None, hyper=None):
    assert isinstance(paths, dict), 'The paths parameter expected a dictionnay but other type is provided'
    assert isinstance(hyper, dict), 'The hyper parameter expected a dictionnay but other type is provided'
    tf.reset_default_graph()
    with tf.gfile.GFile(paths['optimized_pb_path'], 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())
    G = tf.Graph()

    with tf.Session(graph=G) as sess:
        _ = tf.import_graph_def(graph_def_optimized, return_elements=[conserve_nodes[0]])  # note: this line can really clean all input_pipeline/or input what only is necessary
        print('Operations in Optimized Graph:')
        print([op.name for op in G.get_operations()])
        X = G.get_tensor_by_name('import/' + 'input_ph:0')
        y = G.get_tensor_by_name('import/' + 'model/decoder/logits/relu:0')
        do = G.get_tensor_by_name('import/' + 'dropout_ph:0')
        tf.summary.FileWriter(paths['working_dir'] + 'tb/after_optimize', sess.graph)
        y = sess.run(y, feed_dict={X: inputs, do: 1})
        # note: 1.throw up OpenMP error on Mac.
        print(y.shape)


if __name__ == '__main__':
    conserve_nodes = [
            'model/decoder/logits/relu',
        ]
    graph_def_dir = './logs/2019_10_13_bs300_ps72_lr0.0001_cs5_nc80_do0.1_act_leaky_aug_True/hour15/'

    paths = {
        'working_dir': graph_def_dir,
        'ckpt_dir': graph_def_dir + 'ckpt/',
        'ckpt_path': graph_def_dir + 'ckpt/step23192',
        'save_pb_dir': graph_def_dir + 'pb/',
        'save_pb_path': graph_def_dir + 'pb/frozen_step23192.pb',
        'optimized_pb_dir': graph_def_dir + 'optimize/',
        'optimized_pb_path': graph_def_dir + 'optimize/optimized.pb',
        'tflite_pb_dir': graph_def_dir + 'tflite/',
        'data_dir': './data/72/',
        'rlt_dir': graph_def_dir + 'rlt/',
        'GPU': 0,
    }

    hyperparams = {
        'patch_size': 72,
        'batch_size': 300,  # Xlearn < 20, Unet < 20 saturate GPU memory
        'nb_epoch': 100,
        'nb_batch': None,
        'conv_size': 5,
        'nb_conv': 80,
        'learning_rate': 1e-4,  # should use smaller learning rate when decrease batch size
        'dropout': 0.1,
        'device_option': 'specific_gpu:1',
        'augmentation': True,
        'activation': 'leaky',
        'save_step': 1000,
    }

    freeze_ckpt_for_inference(paths=paths, hyper=hyperparams)  #there's still some residual nodes
    optimize_curve_for_inference(paths=paths)  #clean residual nodes
    inputs = np.zeros((hyperparams['batch_size'], hyperparams['patch_size'], hyperparams['patch_size'], 1))
    inference(inputs=inputs, paths=paths, hyper=hyperparams)








