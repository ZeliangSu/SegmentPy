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
    print('norm_w, norm_pert: {}, {}'.format(norm_w, norm_pert))
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
        # print('leap: {}'.format(leap))
        # initial = sess.run(weight)
        # print('\n{} init: {}'.format(name, initial))
        # assign = sess.run(tf.assign(weight, value))
        # print('\n{} Assigned: {}'.format(name, assign))
        after = sess.run(tf.assign(weight, value + leap))
        # print('\nAfter: {}'.format(after))
    except Exception as e:
        print('initial shape: \n', sess.run(value).shape)
        print('leap shape: \n', leap.shape)
        print('\nError threw while trying to move weight: {}'.format(name))
        print(e)


def feed_forward(sess, graph, state, direction_2D, xcoord, ycoord, inputs, outputs, comm=None):
    '''return loss and acc'''
    assert isinstance(direction_2D, list), 'dir should be list'
    assert isinstance(xcoord, float), 'xcoord should be float'
    assert isinstance(ycoord, float), 'ycoord should be float'
    # print('inputs: {}'.format(inputs))
    # print('outputs: {}'.format(outputs))
    # print('xcoord:', xcoord)
    sess.run([tf.local_variables_initializer()])  # note: should initialize here otherwise it will keep cumulating for the average
    # print(sess.run(tf.local_variables()))  #note: uncomment this line to see if local_variables(metrics/loss/count...) are well init
    # change state in the neural network
    new_logits = graph.get_tensor_by_name('model/MLP/logits:0')
    loss_tensor = graph.get_tensor_by_name('metrics/loss/value:0')
    acc_tensor = graph.get_tensor_by_name('metrics/acc/value:0')
    new_loss_update_op = graph.get_operation_by_name('metrics/loss/update_op')
    new_acc_update_op = graph.get_operation_by_name('metrics/acc/update_op')
    new_input_ph = graph.get_tensor_by_name('input_ph:0')
    new_output_ph = graph.get_tensor_by_name('output_ph:0')
    new_BN_ph = graph.get_tensor_by_name('BN_phase:0')

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
                                              new_BN_ph: True,
                                              #note: (TF1.14)WTF? here should be True while producing loss-landscape
                                              #fixme: should check if the mov_avg/mov_std/beta/gamma change
                                              })
    if comm is not None:
        if comm.Get_rank() != 0:
            comm.send(1, dest=0, tag=tag_compute)

    # print('lss:{}, acc:{}, predict:{}'.format(loss, acc, np.mean(new_log)))
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
                  'zval': zval.ravel()}
                 ).to_csv(out_path, index=False)


def clean(array):
    assert isinstance(array, np.ndarray)
    array[np.where(array == np.nan)] = 1e-9
    # array[np.where(array == 0)] = 1e-9
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
        # print(sess.run(tf.local_variables()))  #note: local_variables include metrics/acc/total; metrics/loss/count; metrics/acc/count...
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

    # print('\nlss', tmp_loss)
    # print('\nacc', tmp_acc)
    return tmp_loss.astype(np.float32), tmp_acc.astype(np.float32)

l_steps = [str(i * 1000) for i in range(10)]
l_ckpts = ['./dummy/ckpt/step{}'.format(i) for i in l_steps]
l_states = [get_state(_c) for _c in l_ckpts]
l_directions = [normalize_state(get_random_state(_s), _s) for _s in l_states]
l_directions_bis = [normalize_state(get_random_state(_s), _s) for _s in l_states]

# compute l2-norm and cos
l_angles = []
for i, (dict1, dict2) in enumerate(zip(l_directions, l_directions_bis)):
    tmp = {}
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        assert k1 == k2, 'Found different weights names'
        tmp[k1] = cosine(v1, v2)
    l_angles.append(tmp)

l_L2norm = []
for i, (dict1, dict2) in enumerate(zip(l_directions, l_directions_bis)):
    tmp = {}
    for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
        assert k1 == k2, 'Found different weights names'
        tmp[k1] = euclidean(v1, v2)
    l_angles.append(tmp)

# write cos and l2-norm to xlsw
for i, angle in enumerate(l_angles):
    with pd.ExcelWriter('./dummy/cosin.xlsx', engine='xlsxwriter') as writer:
        angle['step{}'.format(i * 1000)].to_excel(writer, index=False, header=False)

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


communicator = MPI.COMM_WORLD
rank = communicator.Get_rank()
nb_process = communicator.Get_size()

total_computation = xm.size
remainder = total_computation % (nb_process - 1)  # master node manage remainder
bus_per_rank = total_computation // (nb_process - 1)  # sub-nodes compute others

print('MPI_version', MPI.get_vendor())
print('This rank is:', rank)
print('nb_process', nb_process)
# **************************************************************************************************** I'm a Barrier
communicator.Barrier()

#note: numpy use Send/Recv, list/dict use send/recv
tag_compute = 0
tag_end = 99

for _step, _ckpt, _state, _dir, _dir_bis in tqdm(zip(l_steps, l_ckpts, l_states, l_directions, l_directions_bis)
                                                 , desc='Checkpoint'):
    # **************************************************************************************************** I'm a Barrier
    communicator.Barrier()
    # fixme: put it somewhere else

    # init placeholder
    if rank == 0:
        shared_lss = np.empty(xm.shape, dtype=np.float32).ravel()
        shared_acc = np.empty(xm.shape, dtype=np.float32).ravel()
        loss_ph = np.empty(bus_per_rank, dtype=np.float32)
        acc_ph = np.empty(bus_per_rank, dtype=np.float32)
        _dir1_ph = None
        _dir2_ph = None
        xm_ph = None
        ym_ph = None

        pbar = tqdm(total=total_computation)
        update_msg = None



    else:
        shared_lss = None
        shared_acc = None
        loss_ph = None
        acc_ph = None
        xm_ph = np.empty(bus_per_rank, dtype=np.float32)
        ym_ph = np.empty(bus_per_rank, dtype=np.float32)
        update_msg = None


    # **************************************************************************************************** I'm a Barrier
    communicator.Barrier()


    # from 0 send buses to sub-process
    if rank == 0:
        print('\n****Start scattering')
        try:
            # send order
            count = 0
            remaining = nb_process - 1
            for _rank in tqdm(range(1, nb_process)):
                communicator.send(_dir, dest=_rank, tag=31)
                communicator.send(_dir_bis, dest=_rank, tag=32)
                communicator.Send(xm.ravel()[(_rank - 1) * bus_per_rank: _rank * bus_per_rank], dest=_rank, tag=44)
                communicator.Send(ym.ravel()[(_rank - 1) * bus_per_rank: _rank * bus_per_rank], dest=_rank, tag=55)
                count += 1

            print('Rank {} sent successfully'.format(rank))


        except Exception as e:
            print('While sending buses, \nRank {} throws error: {}'.format(rank, e))
            break

        try:
            _loss, _acc = feed_forward_MP(ckpt_path=_ckpt,
                                          state=_state,
                                          direction_2D=[_dir, _dir_bis],
                                          x_mesh=xm.ravel()[-remainder:],
                                          y_mesh=ym.ravel()[-remainder:],
                                          comm=communicator
                                          )

            shared_lss[-remainder:] = _loss
            shared_acc[-remainder:] = _acc

            while remaining > 0:
                s = MPI.Status()
                communicator.Probe(status=s)
                if s.tag == tag_compute:
                    update_msg = communicator.recv(tag=tag_compute)
                    pbar.update(1)
                elif s.tag == tag_end:
                    update_msg = communicator.recv(tag=tag_end)
                    remaining -= 1
                    print('remaining: {}', remaining)

            print('Rank 0 out of while loop')

        except Exception as e:
            print('While computing, \nRank {} throws error: {}'.format(rank, e))

    else:
        try:
            # receive
            _dir1_ph = communicator.recv(source=0, tag=31)
            _dir2_ph = communicator.recv(source=0, tag=32)
            communicator.Recv(xm_ph, source=0, tag=44)
            communicator.Recv(ym_ph, source=0, tag=55)
            print('Rank {} received successfully'.format(rank))
        except Exception as e:
            print('While sending buses, \nRank {} throws error: {}'.format(rank, e))
            break

        try:
            # compute
            _loss, _acc = feed_forward_MP(ckpt_path=_ckpt,
                                          state=_state,
                                          direction_2D=[_dir1_ph, _dir2_ph],
                                          x_mesh=xm_ph,
                                          y_mesh=ym_ph,
                                          comm=communicator
                                          )
            communicator.send(1, dest=0, tag=tag_end)
        except Exception as e:
            print('While computing, \nRank {} throws error: {}'.format(rank, e))


    # **************************************************************************************************** I'm a Barrier
    print('Hello!')
    communicator.Barrier()
    print('\n****Start gathering')

    # Send back and Gathering
    if rank == 0:
        try:
            # gathering
            for _rank in tqdm(range(1, nb_process)):
                communicator.Recv(loss_ph, source=_rank, tag=91)
                communicator.Recv(acc_ph, source=_rank, tag=92)
                print('Received from rank {} successfully'.format(_rank))
                shared_lss[(_rank - 1) * bus_per_rank: _rank * bus_per_rank] = loss_ph
                shared_acc[(_rank - 1) * bus_per_rank: _rank * bus_per_rank] = acc_ph
        except Exception as e:
            print('While gathering buses, \nRank {} throws error: {}'.format(rank, e))

    else:
        try:
            # send back
            communicator.Send(_loss, dest=0, tag=91)
            communicator.Send(_acc, dest=0, tag=92)
            print('Rank {} sent successfully'.format(rank))
        except Exception as e:
            print('While gathering buses, \nRank {} throws error: {}'.format(rank, e))

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
        # csv_interp(xm, ym, shared_lss, './dummy/paraview_lss_step{}.csv'.format(_step))
        # csv_interp(xm, ym, shared_acc, './dummy/paraview_lss_step{}.csv'.format(_step))

    # **************************************************************************************************** I'm a Barrier
    communicator.Barrier()