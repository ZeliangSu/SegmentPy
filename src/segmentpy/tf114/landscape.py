from mpi4py import MPI
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from input import _one_hot, _inverse_one_hot
from layers import customized_softmax_np
from util import print_nodes_name_shape
import copy
import pandas as pd
from scipy.interpolate import interp2d
from scipy.spatial.distance import cosine, euclidean
from PIL import Image

import logging
import log
logger = log.setup_custom_logger('root')
logger.setLevel(logging.ERROR)

tag_pbar = 0
tag_data = 255

import os
# prevent GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 1})

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
        # print_nodes_name_shape(tf.get_default_graph())
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
        print(e)


def csv_interp(x_mesh, y_mesh, metrics_tensor, out_path, interp_scope=5):
    new_xmesh = np.linspace(np.min(x_mesh), np.max(x_mesh), interp_scope * x_mesh.shape[0])
    new_ymesh = np.linspace(np.min(y_mesh), np.max(y_mesh), interp_scope * x_mesh.shape[1])
    newxx, newyy = np.meshgrid(new_xmesh, new_ymesh)

    # interpolation
    interpolation = interp2d(x_mesh, y_mesh, metrics_tensor, kind='cubic')
    zval = interpolation(new_xmesh, new_ymesh)
    pd.DataFrame({'xcoord': newxx.ravel(),
                  'ycoord': newyy.ravel(),
                  'zval': zval.ravel()}
                 ).to_csv(out_path, index=False)


def clean(array):
    assert isinstance(array, np.ndarray)
    array[np.where(array == np.nan)] = 1e-9
    # array[np.where(array == 0)] = 1e-9
    array[np.where(array == np.inf)] = 1e9
    array[np.where(array == -np.inf)] = -1e9
    return array


def feed_forward(sess, graph, state, direction_2D, xcoord, ycoord, inputs, outputs, comm=None):
    '''return loss and acc'''
    assert isinstance(direction_2D, list), 'dir should be list'
    assert isinstance(xcoord, float), 'xcoord should be float'
    assert isinstance(ycoord, float), 'ycoord should be float'
    logger.debug('inputs: {}'.format(inputs))
    logger.debug('outputs: {}'.format(outputs))
    logger.debug('xcoord:', xcoord)
    sess.run([tf.local_variables_initializer()])  # note: should initialize here otherwise it will keep cumulating for the average
    
    # print(sess.run(tf.local_variables()))  #note: uncomment this line to see if local_variables(metrics/loss/count...) are well init
    # tf.summary.FileWriter('./dummy/debug/', sess.graph)
    # change state in the neural network
    
    ###############################################################################################
    # todo: this part should be automatic for different model
    new_logits = graph.get_tensor_by_name('LRCS/decoder/logits/identity:0')
    loss_tensor = graph.get_tensor_by_name('train_metrics/ls_train/value:0')
    acc_tensor = graph.get_tensor_by_name('train_metrics/acc_train/value:0')
    new_loss_update_op = graph.get_operation_by_name('train_metrics/ls_train/update_op')
    new_acc_update_op = graph.get_operation_by_name('train_metrics/acc_train/update_op')
    new_input_ph = graph.get_tensor_by_name('input_pipeline_train/IteratorGetNext:0')
    new_do_ph = graph.get_tensor_by_name('dropout_prob:0')
    new_BN_ph = graph.get_tensor_by_name('BN_phase:0')  # todo:
    new_output_ph = graph.get_tensor_by_name('input_pipeline_train/IteratorGetNext:1')
    # print(tf.shape(new_input_ph))
    # print(tf.shape(new_output_ph))
    
    ###############################################################################################
    # print('inputs avg: {}, outputs avg: {}'.format(np.mean(inputs), np.mean(outputs)))

    dx = {k: xcoord * v for k, v in direction_2D[0].items()}  # step size * direction x
    dy = {k: ycoord * v for k, v in direction_2D[1].items()}  # step size * direction y
    change = {k: _dx + _dy for (k, _dx), (_, _dy) in zip(dx.items(), dy.items())}
    
    # calculate the perturbation
    for k, v in state.items():
        move_state(sess, name=k, value=v, leap=change[k])

    # feed forward with batches
    for repeat in range(2):
        #note: (TF intrisic: at least 2 times, or loss/acc 0.0)should iterate at least several time, Or loss=acc=0, since there's a counter for the average
        new_log, loss_ff, acc_ff, _, _ = sess.run([new_logits, loss_tensor, acc_tensor, new_acc_update_op, new_loss_update_op],
                                   feed_dict={new_input_ph: inputs,
                                              new_output_ph: outputs,
                                              new_do_ph: 1.0,  #todo:
                                              new_BN_ph: True,  # todo:
                                              #note: (TF1.14)WTF? here should be True while producing loss-landscape
                                              #fixme: should check if the mov_avg/mov_std/beta/gamma change
                                              })
    if comm is not None:
        if comm.Get_rank() != 0:
            comm.send(1, dest=0, tag=tag_pbar)
            comm.send([xcoord, ycoord, loss_ff, acc_ff], dest=0, tag=tag_data)
            
    new_log = _inverse_one_hot(customized_softmax_np(new_log))
    logger.debug('lss:{}, acc:{}, predict:{}'.format(loss_ff, acc_ff, np.mean(new_log)))
    return loss_ff, acc_ff


def feed_forward_MP(ckpt_path, state, direction_2D, x_mesh, y_mesh, comm=None):
    assert isinstance(direction_2D, list)  # list of dicts
    assert isinstance(x_mesh, np.ndarray) and x_mesh.ndim == 1
    assert isinstance(y_mesh, np.ndarray) and y_mesh.ndim == 1

    loader = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
    tmp_loss, tmp_acc = np.zeros(x_mesh.size), np.zeros(x_mesh.size)

    with tf.Session() as sess:
        sess.run([tf.local_variables_initializer()])
        # print(sess.run(tf.local_variables()))  #note: local_variables include metrics/acc/total; metrics/loss/count; metrics/acc/count...
        loader.restore(sess, ckpt_path)
        graph = tf.get_default_graph()
        # print_nodes_name_shape(graph)
        for i in range(x_mesh.size):
            # todo: don't use the training data here
            inputs = np.asarray([np.asarray(Image.open('./raw/0.tif'))[i*50: 512 + i*50, i*50:512 + i*50] for i in range(8)]).reshape((8, 512, 512, 1))  #todo: automatization with test data
            outputs = _one_hot(np.asarray([np.asarray(Image.open('./raw/0_label.tif'))[i*50: 512 + i*50, i*50:512 + i*50] for i in range(8)]).reshape((8, 512, 512, 1)))
            tmp_loss[i], tmp_acc[i] = feed_forward(sess=sess,
                                           graph=graph,
                                           state=state,
                                           direction_2D=direction_2D,
                                           xcoord=float(x_mesh[i]),   #note: can debug with 0.0
                                           ycoord=float(y_mesh[i]),   #note: can debug with 0.0
                                           inputs=inputs,
                                           outputs=outputs,
                                           comm=comm)
            print('\nlss', tmp_loss[i])
            print('\nacc', tmp_acc[i])
    return tmp_loss.astype(np.float32), tmp_acc.astype(np.float32)

def landscape():
    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    nb_process = communicator.Get_size()
    
    l_steps = [0, 28219]
    print('***********************************************************************')
    if rank == 0:
        print('***********************************************************************')
        l_ckpts = ['./logs/model_20200219_hour8/ckpt/step{}'.format(i) for i in l_steps]
        l_states = [get_state(_c) for _c in l_ckpts]
        l_directions = [normalize_state(get_random_state(_s), _s) for _s in l_states]
        l_directions_bis = [normalize_state(get_random_state(_s), _s) for _s in l_states]
    
    #########################################################
    # # compute l2-norm and cos
    # l_angles = []
    # for i, (dict1, dict2) in enumerate(zip(l_directions, l_directions_bis)):
    #     tmp = {}
    #     for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
    #         assert k1 == k2, 'Found different weights names'
    #         tmp[k1] = cosine(v1, v2)
    #     l_angles.append(tmp)
    #
    # l_L2norm = []
    # for i, (dict1, dict2) in enumerate(zip(l_directions, l_directions_bis)):
    #     tmp = {}
    #     for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
    #         assert k1 == k2, 'Found different weights names'
    #         tmp[k1] = euclidean(v1, v2)
    #     l_angles.append(tmp)
    #
    # # write cos and l2-norm to xlsw
    # for i, angle in enumerate(l_angles):
    #     with pd.ExcelWriter('./dummy/cosin.xlsx', engine='xlsxwriter') as writer:
    #         angle['step{}'.format(28219)].to_excel(writer, index=False, header=False)
    #########################################################
    
    # create direction and surface .h5
    x_min, x_max, x_nb = -1, 1, 51
    y_min, y_max, y_nb = -1, 1, 51
    xcoord = np.linspace(x_min, x_max, x_nb).ravel()
    ycoord = np.linspace(y_min, y_max, y_nb).ravel()
    xm, ym = np.meshgrid(xcoord, ycoord)
    xm = xm.astype(np.float32)
    ym = ym.astype(np.float32)
    
    # calculate loss/acc for each point on the surface (try first with only for loop)
    # start feeding
    
    total_computation = remaining = xm.size
    remainder = total_computation % (nb_process - 1)  # master node manage remainder
    bus_per_rank = total_computation // (nb_process - 1)  # sub-nodes compute others
    
    print('MPI_version', MPI.get_vendor())
    print('This rank is:', rank)
    print('nb_process', nb_process)
    # **************************************************************************************************** I'm a Barrier
    communicator.Barrier()
    print('***********************************************************************')
    #note: numpy use Send/Recv (don't use this here), list/dict use send/recv    
    for a, _step in tqdm(enumerate(l_steps), desc='Checkpoint'):
        # **************************************************************************************************** I'm a Barrier
        communicator.Barrier()
        print('***********************************************************************')
        # init placeholder
        if rank == 0:
            shared_lss = np.empty(xm.shape, dtype=np.float32).ravel()
            shared_acc = np.empty(xm.shape, dtype=np.float32).ravel()
            pbar = tqdm(total=total_computation)
            update_msg = None
            print('***********************************************************************')
    
        else:
            xm_ph = None
            ym_ph = None
            _dir1_ph = None
            _dir2_ph = None
            _ckpt_ph = None
            _state_ph = None
            update_msg = None
    
        # from 0 send buses to sub-process
        if rank == 0:
            print('***********************************************************************')
            try:
                # send order
                for _rank in tqdm(range(1, nb_process)):
                    communicator.send(l_directions[a], dest=_rank, tag=1)
                    communicator.send(l_directions_bis[a], dest=_rank, tag=2)
                    communicator.send(l_states[a], dest=_rank, tag=3)
                    communicator.send(l_ckpts[a], dest=_rank, tag=4)
                    if _rank <= remainder:
                        communicator.send([xm.ravel()[(_rank - 1) * (bus_per_rank + 1): _rank * (bus_per_rank + 1)]], dest=_rank, tag=55)
                        communicator.send([ym.ravel()[(_rank - 1) * (bus_per_rank + 1): _rank * (bus_per_rank + 1)]], dest=_rank, tag=66)
                    else:
                        communicator.send([xm.ravel()[_rank * bus_per_rank + remainder: _rank * bus_per_rank + remainder]], dest=_rank, tag=55)
                        communicator.send([ym.ravel()[_rank * bus_per_rank + remainder: _rank * bus_per_rank + remainder]], dest=_rank, tag=66)
                    
                print('Rank {} sent successfully'.format(rank))
    
            except Exception as e:
                print('While sending buses, \nRank {} throws error: {}'.format(rank, e))
    
            try:
                print('yoh')
                while True:
                    s = MPI.Status()
                    communicator.Probe(status=s)
                    print(remaining)
                    print(s.tag)
                    if s.tag == tag_pbar:
                        update_msg = communicator.recv(tag=tag_pbar)
                        xc, yc, lss, acc = communicator.recv(tag=tag_data)
                        shared_acc[int(xc), int(yc)] = acc
                        shared_lss[int(xc), int(yc)] = lss
                        pbar.update(1)
                        remaining -= 1
    
                print('Rank 0 out of while loop')
    
            except Exception as e:
                print('While listening, \nRank {} throws error: {}'.format(rank, e))
    
        else:
            try:
                # receive
                _dir1_ph = communicator.recv(source=0, tag=1)
                print('dir1', type(_dir1_ph))
                _dir2_ph = communicator.recv(source=0, tag=2)
                print('dir2', type(_dir2_ph))
                _state_ph = communicator.recv(source=0, tag=3)
                print('state', type(_state_ph))
                _ckpt_ph = communicator.recv(source=0, tag=4)
                print('ckpt', type(_ckpt_ph))
                
                xm_ph = communicator.recv(source=0, tag=55)[0]
                print('xm_ph', type(xm_ph))
                ym_ph = communicator.recv(source=0, tag=66)[0]
                print('ym_ph', type(ym_ph))          
                print('Rank {} received successfully'.format(rank))
            except Exception as e:
                print('While receiving buses, \nRank {} throws error: {}'.format(rank, e))
    
            try:
                # compute
                _loss, _acc = feed_forward_MP(ckpt_path=_ckpt_ph,
                                              state=_state_ph,
                                              direction_2D=[_dir1_ph, _dir2_ph],
                                              x_mesh=xm_ph,
                                              y_mesh=ym_ph,
                                              comm=communicator
                                              )
            except Exception as e:
                print('While computing, \nRank {} throws error: {}'.format(rank, e))
    
    
        # **************************************************************************************************** I'm a Barrier
        communicator.Barrier()
        # save and plot
        if rank == 0:
            shared_lss = shared_lss.reshape(xm.shape)
            shared_acc = shared_acc.reshape(xm.shape)
    
            # take the log for loss
            shared_lss = clean(shared_lss)
            # shared_lss = np.log(shared_lss)
    
            # plot results
            # fig, ax1 = plt.subplots(1)
            # cs1 = ax1.contour(xm, ym, shared_lss)
            # plt.clabel(cs1, inline=1, fontsize=10)
            # fig2, ax2 = plt.subplots(1)
            # cs2 = ax2.contour(xm, ym, shared_acc)
            # plt.clabel(cs2, inline=1, fontsize=10)
            # plt.show()
    
            pd.DataFrame(shared_lss).to_csv('./dummy/lss_step{}.csv'.format(_step), index=False)
            pd.DataFrame(shared_acc).to_csv('./dummy/acc_step{}.csv'.format(_step), index=False)
            csv_interp(xm, ym, shared_lss, './dummy/paraview_lss_step{}.csv'.format(_step))
            csv_interp(xm, ym, shared_acc, './dummy/paraview_lss_step{}.csv'.format(_step))

if __name__ == '__main__':
    landscape()
    
