from mpi4py import MPI
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from util import print_nodes_name_shape
import copy
import pandas as pd
from scipy.interpolate import interp2d
from scipy.spatial.distance import cosine, euclidean

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level




# define some constants
tag_compute = 0
tag_end = 99

#note: state includes weights, bias, BN
def get_diff_state(state1, state2):
    assert isinstance(state1, dict)
    assert isinstance(state2, dict)
    return {k: v2 - v1 for (k, _), (v1, v2) in zip(state1.items(), state2.items())}


def get_random_state(state):
    assert isinstance(state, dict)
    return {k: np.random.randn(*v.shape) for k, v in state.items()}


def get_state(ckpt_path):
    tf.reset_default_graph()
    loader = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
    state = {}
    with tf.Session() as sess:
        loader.restore(sess, ckpt_path)
        print_nodes_name_shape(tf.get_default_graph())
        ckpt_weight_names = []
        for node in tf.get_default_graph().as_graph_def().node:
            if node.name.endswith('w1') or \
                    node.name.endswith('b1') or \
                    node.name.endswith('beta') or \
                    node.name.endswith('gamma'):
                ckpt_weight_names.append(node.name + ':0')

        # get weights/bias
        for k in ckpt_weight_names:
            v = sess.run(k)
            state[k] = v
    return state


def _normalize_direction(perturbation, weight):
    assert isinstance(perturbation, np.ndarray)
    assert isinstance(weight, np.ndarray)
    norm_w = np.linalg.norm(weight)
    norm_pert = np.linalg.norm(perturbation)
    logger.debug('norm_w, norm_pert: {}, {}'.format(norm_w, norm_pert))
    perturbation *= norm_w / (norm_pert)
    return perturbation, weight


def normalize_state(directions, state):
    assert isinstance(directions, dict)
    assert isinstance(state, dict)
    backup = copy.deepcopy(directions)
    for (d_name, direct), (name, weight) in zip(backup.items(), state.items()):
        _normalize_direction(direct, weight)
    return backup


def move_state(sess, name, value, leap):
    # if ('beta' not in name) and ('gamma' not in name):
    weight = sess.graph.get_tensor_by_name(name)
    try:
        logger.debug('leap: {}'.format(leap))
        initial = sess.run(weight)
        logger.debug('\n{} init: {}'.format(name, initial))
        assign = sess.run(tf.assign(weight, value))
        logger.debug('\n{} Assigned: {}'.format(name, assign))
        after = sess.run(tf.assign(weight, value + leap))
        logger.debug('\nAfter: {}'.format(after))
    except Exception as e:
        logger.debug('initial shape: \n', sess.run(value).shape)
        logger.debug('leap shape: \n', leap.shape)
        logger.debug('\nError threw while trying to move weight: {}'.format(name))
        logger.debug(e)


def feed_forward(sess, graph, state, direction_2D, xcoord, ycoord, inputs, outputs, comm=None):
    '''return loss and acc'''
    assert isinstance(direction_2D, list), 'dir should be list'
    assert isinstance(xcoord, float), 'xcoord should be float'
    assert isinstance(ycoord, float), 'ycoord should be float'

    logger.debug('inputs: {}'.format(inputs))
    logger.debug('outputs: {}'.format(outputs))
    logger.debug('xcoord:', xcoord)
    sess.run([tf.local_variables_initializer()])  # note: should initialize here otherwise it will keep cumulating for the average
    logger.debug(sess.run(tf.local_variables()))  #note: uncomment this line to see if local_variables(metrics/loss/count...) are well init
    # change state in the neural network
    new_logits = graph.get_tensor_by_name('model/MLP/logits:0')
    loss_tensor = graph.get_tensor_by_name('metrics/loss/value:0')
    acc_tensor = graph.get_tensor_by_name('metrics/acc/value:0')
    new_loss_update_op = graph.get_operation_by_name('metrics/loss/update_op')
    new_acc_update_op = graph.get_operation_by_name('metrics/acc/update_op')
    new_input_ph = graph.get_tensor_by_name('input_ph:0')
    new_output_ph = graph.get_tensor_by_name('output_ph:0')
    new_BN_ph = graph.get_tensor_by_name('BN_phase:0')

    logger.debug('inputs avg: {}, outputs avg: {}'.format(np.mean(inputs), np.mean(outputs)))

    dx = {k: xcoord * v for k, v in direction_2D[0].items()}  # step size * direction x
    dy = {k: ycoord * v for k, v in direction_2D[1].items()}  # step size * direction y
    change = {k: _dx + _dy for (k, _dx), (_, _dy) in zip(dx.items(), dy.items())}
    # calculate the perturbation
    for k, v in state.items():
        move_state(sess, name=k, value=v, leap=change[k])

    loss_ff, acc_ff, new_log = None, None, None
    # feed forward with batches
    for repeat in range(2):
        #note: (TF intrisic: at least 2 times, or loss/acc 0.0)should iterate at least several time, Or loss=acc=0, since there's a counter for the average
        new_log, loss_ff, acc_ff, _, _ = sess.run([new_logits, loss_tensor, acc_tensor, new_acc_update_op, new_loss_update_op],
                                   feed_dict={new_input_ph: inputs,
                                              new_output_ph: outputs,
                                              new_BN_ph: True,
                                              #note: (TF1.14)WTF? here should be True while producing loss-landscape
                                              #fixme: should check if the mov_avg/mov_std/beta/gamma change
                                              })
    if comm is not None:
        if comm.Get_rank() != 0:
            comm.send(1, dest=0, tag=tag_compute)

    logger.debug('lss:{}, acc:{}, predict:{}'.format(loss_ff, acc_ff, np.mean(new_log)))
    return loss_ff, acc_ff


def csv_interp(x_mesh, y_mesh, metrics_tensor, out_path, interp_scope=5):
    new_xmesh = np.linspace(np.min(x_mesh), np.max(x_mesh), interp_scope * x_mesh.shape[0])
    new_ymesh = np.linspace(np.min(y_mesh), np.max(y_mesh), interp_scope * x_mesh.shape[1])
    newxx, newyy = np.meshgrid(new_xmesh, new_ymesh)

    # interpolation
    interpolation = interp2d(x_mesh, y_mesh, metrics_tensor, kind='cubic')
    zval = interpolation(new_xmesh, new_ymesh)
    pd.DataFrame({'xcoord': newxx.ravel(),
                  'ycoord': newyy.ravel(),
                  'zval': zval.ravel()}).to_csv(out_path, index=False)


def clean(array, clean_zeros=False):
    assert isinstance(array, np.ndarray)
    array[np.where(array == np.nan)] = 1e-9
    if clean_zeros:
        array[np.where(array == 0)] = 1e-9
    array[np.where(array == np.inf)] = 1e9
    array[np.where(array == -np.inf)] = -1e9
    return array


def feed_forward_MP(ckpt_path, state, direction_2D, x_mesh, y_mesh, comm=None):
    assert isinstance(direction_2D, list)  # list of dicts
    assert isinstance(x_mesh, np.ndarray) and x_mesh.ndim == 1
    assert isinstance(y_mesh, np.ndarray) and y_mesh.ndim == 1

    loader = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
    tmp_loss, tmp_acc = np.zeros(x_mesh.size), np.zeros(x_mesh.size)

    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer()])

        logger.debug(sess.run(tf.local_variables()))  #note: local_variables include metrics/acc/total; metrics/loss/count; metrics/acc/count...
        loader.restore(sess, ckpt_path)
        graph = tf.get_default_graph()

        # print_nodes_name_shape(graph)
        for i in range(x_mesh.size):
            inputs = np.random.randn(8, 20, 20, 1) + np.ones((8, 20, 20, 1)) * 8  # noise + avg:8
            outputs = np.ones((8, 20, 20, 1)) * 64   # avg: 9
            tmp_loss[i], tmp_acc[i] = feed_forward(sess=sess,
                                           graph=graph,
                                           state=state,
                                           direction_2D=direction_2D,
                                           xcoord=float(x_mesh[i]),   #note: can debug with 0.0
                                           ycoord=float(y_mesh[i]),   #note: can debug with 0.0
                                           inputs=inputs,
                                           outputs=outputs,
                                           comm=comm)

