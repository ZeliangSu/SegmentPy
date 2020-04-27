import tensorflow as tf
import numpy as np
import os
from util import check_N_mkdir
from itertools import product
from PIL import Image
from tqdm import tqdm
from input import _minmaxscalar
from util import print_nodes_name, get_list_fnames
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
from layers import customized_softmax_np
import argparse
import re

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)

tag_compute = 1002


# argparser
parser = argparse.ArgumentParser('main.py')
parser.add_argument('-ckpt', '--ckpt_path', type=str, metavar='', required=True, help='.meta path')
parser.add_argument('-raw', '--raw_path', type=str, metavar='', required=True, help='raw tomograms folder path')
parser.add_argument('-pred', '--pred_path', type=str, metavar='', required=True, help='where to put the segmentation')
parser.add_argument('-bs', '--batch_size', type=int, metavar='', required=False, const=8, help='identical as the trained model')
parser.add_argument('-ws', '--window_size', type=int, metavar='', required=False, const=512, help='identical as the trained model')
args = parser.parse_args()
print(args)



class reconstructor_V2_reg():
    def __init__(self, image_size, patch_size, stride):
        self.i_h, self.i_w = image_size[:2]  # e.g. (a, b)
        # todo: p_h and p_w can be generalize to the case that they are different
        self.p_h, self.p_w = patch_size, patch_size  # e.g. (a, b)
        self.stride = stride
        self.result = np.zeros((self.i_h, self.i_w))
        self.n_h = (self.i_h - self.p_h) // self.stride + 1
        self.n_w = (self.i_w - self.p_w) // self.stride + 1

    def add_batch(self, batch, start_id):
        id_list = np.arange(start_id, start_id + batch.shape[0], 1)
        for i, id in enumerate(id_list):
            b = id // self.n_h
            a = id % self.n_h
            self.result[a * self.stride: a * self.stride + self.p_h, b * self.stride: b * self.stride + self.p_w] += np.squeeze(batch[i])

    def reconstruct(self):
        print('***Reconostructing')
        for i in range(self.i_h):
            for j in range(self.i_w):
                self.result[i, j] /= float(min(i + self.stride, self.p_h, self.i_h - i) *
                                   min(j + self.stride, self.p_w, self.i_w - j))

    def get_reconstruction(self):
        return self.result

    def get_nb_patch(self):
        return self.n_h, self.n_w


class reconstructor_V2_cls():
    def __init__(self, image_size, patch_size, stride, nb_class):
        self.i_h, self.i_w = image_size[:2]  # e.g. (a, b)
        self.p_h, self.p_w = patch_size, patch_size  # e.g. (a, b)
        self.nb_cls = nb_class
        self.stride = stride
        self.result = np.zeros((self.i_h, self.i_w, self.nb_cls))
        self.n_h = (self.i_h - self.p_h) // self.stride + 1
        self.n_w = (self.i_w - self.p_w) // self.stride + 1

    def add_batch(self,
                  batch,  # (B, H, W, C)
                  start_id):
        id_list = np.arange(start_id, start_id + batch.shape[0], 1)
        for i, id in enumerate(id_list):
            b = id // self.n_h
            a = id % self.n_h
            tmp = np.argmax(batch[i], axis=2)  # (H, W)
            # nb_classes = len(np.unique(tmp))  # might be a bug
            out = []
            for j in range(self.nb_cls):
                onehot = np.zeros(tmp.shape)
                onehot[np.where(tmp == j)] = 1
                out.append(np.expand_dims(onehot, axis=2))

            # stack along the last channel
            out = np.concatenate(out, axis=2)  # (H, W, C)
            #
            # print('id:{}'.format(id))
            # print('stt_id:{}'.format(start_id))
            # print('shape:{}'.format(batch.shape))
            # print('a:{}'.format(a))
            # print('b:{}'.format(b))
            self.result[a * self.stride: a * self.stride + self.p_h, b * self.stride: b * self.stride + self.p_w, :] += out

    def reconstruct(self):
        print('***Reconostructing')
        self.result = np.argmax(self.result, axis=2).astype(float)

    def get_reconstruction(self):
        return self.result

    def get_nb_patch(self):
        return self.n_h, self.n_w


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
    assert isinstance(conserve_nodes, list), 'The name of the conserve node should be in a list'

    # clean graph first
    tf.reset_default_graph()

    # freeze ckpt then convert to pb
    new_input = tf.placeholder(tf.float32, shape=[None, hyper['patch_size'], hyper['patch_size'], 1], name='new_input')
    new_BN = tf.placeholder_with_default(False, [], name='new_BN')  #note: it seems like T/F after freezing isn't important

    # load meta graph
    input_map = {
        'input_pipeline_train/IteratorGetNext': new_input,
    }
    try:
        new_dropout = tf.placeholder_with_default(1.0, [], name='new_dropout')
        input_map['dropout_prob'] = new_dropout
        if hyper['batch_normalization']:
            input_map['BN_phase'] = new_BN

        restorer = tf.train.import_meta_graph(
            paths['ckpt_path'] + '.meta' if not paths['ckpt_path'].endswith('.meta') else paths['ckpt_path'],
            input_map=input_map,
            clear_devices=True,
        )
    except Exception as e:
        if hyper['batch_normalization']:
            input_map['BN_phase'] = new_BN

        logger.warning('Error(msg):', e)
        restorer = tf.train.import_meta_graph(
            paths['ckpt_path'] + '.meta' if not paths['ckpt_path'].endswith('.meta') else paths['ckpt_path'],
            input_map=input_map,
            clear_devices=True,
        )

    input_graph_def = tf.get_default_graph().as_graph_def()
    check_N_mkdir(paths['save_pb_dir'])

    # use cpu or gpu
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

    # freeze to pb
    with tf.Session(**config_params) as sess:
        # restore variables
        restorer.restore(sess, paths['ckpt_path'])
        # convert variable to constant
        # todo: verify this convert_variables_to_constants() function if it's correctly working for batch norm
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,
            input_graph_def=input_graph_def,
            output_node_names=conserve_nodes,
        )

        # save to pb
        tf.summary.FileWriter(paths['working_dir'] + 'tb/after_freeze', sess.graph)
        with tf.gfile.GFile(paths['save_pb_path'], 'wb') as f:  # 'wb' stands for write binary
            f.write(output_graph_def.SerializeToString())


def optimize_pb_for_inference(paths=None, conserve_nodes=None):
    assert isinstance(paths, dict), 'The paths parameter expected a dictionnay but other type is provided'
    # clean graph first
    tf.reset_default_graph()
    check_N_mkdir(paths['save_pb_dir'])

    # load protobuff
    with tf.gfile.FastGFile(paths['save_pb_path'], "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # optimize pb
    optimize_graph_def = optimize_for_inference(
        input_graph_def=graph_def,
        input_node_names=['new_input', 'new_BN', 'new_dropout'],
        output_node_names=conserve_nodes,
        placeholder_type_enum=[dtypes.float32.as_datatype_enum,
                               dtypes.bool.as_datatype_enum,
                               dtypes.float32.as_datatype_enum,
                               ]
    )
    with tf.gfile.GFile(paths['optimized_pb_path'], 'wb') as f:
        f.write(optimize_graph_def.SerializeToString())


def inference_recursive_V2(l_input_path=None, conserve_nodes=None, paths=None, hyper=None):
    assert isinstance(conserve_nodes, list), 'conserve nodes should be a list'
    assert isinstance(l_input_path, list), 'inputs is expected to be a list of images for heterogeneous image size!'
    assert isinstance(paths, dict), 'paths should be a dict'
    assert isinstance(hyper, dict), 'hyper should be a dict'

    from mpi4py import MPI
    # prevent GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    communicator = MPI.COMM_WORLD
    rank = communicator.Get_rank()
    nb_process = communicator.Get_size()

    # optimize ckpt to pb for inference
    if rank == 0:
        check_N_mkdir(paths['out_dir'])
        freeze_ckpt_for_inference(paths=paths, hyper=hyper,
                                  conserve_nodes=conserve_nodes)  # there's still some residual nodes
        optimize_pb_for_inference(paths=paths,
                                  conserve_nodes=conserve_nodes)  # clean residual nodes: gradients, td.data.pipeline...
        pbar1 = tqdm(total=len(l_input_path))

    # ************************************************************************************************ I'm a Barrier
    communicator.Barrier()

    # reconstruct volumn
    for i, path in enumerate(l_input_path):
        # ************************************************************************************************ I'm a Barrier
        communicator.Barrier()
        img = np.asarray(Image.open(path))
        n_h = (img.shape[0] - hyper['patch_size']) // hyper['stride'] + 1
        n_w = (img.shape[1] - hyper['patch_size']) // hyper['stride'] + 1
        remaining = n_h * n_w
        nb_img_per_rank = remaining // (nb_process - 1)
        rest_img = remaining % (nb_process - 1)

        if rank == 0:
            pbar2 = tqdm(total=remaining)
            # use reconstructor to not saturate the RAM
            if hyper['mode'] == 'classification':
                reconstructor = reconstructor_V2_cls(img.shape, hyper['patch_size'], hyper['stride'], hyper['nb_classes'])
            else:
                raise NotImplementedError('inference recursive V2 not implemented yet for regression')

            # start gathering batches from other rank
            s = MPI.Status()
            communicator.Probe(status=s)
            while remaining > 0:
                # if s.tag != -1:
                #     print(s.tag)

                if s.tag == tag_compute:
                    # receive outputs
                    core, stt_id, out_batch = communicator.recv(tag=tag_compute)
                    reconstructor.add_batch(out_batch, stt_id)

                    # progress
                    remaining -= out_batch.shape[0]
                    pbar2.update(out_batch.shape[0])

        else:
            if (rank - 1) <= rest_img:
                start_id = (rank - 1) * (nb_img_per_rank + 1)
                id_list = np.arange(start_id, start_id + nb_img_per_rank + 1, 1)
            else:
                start_id = (rank - 1) * nb_img_per_rank + rest_img
                id_list = np.arange(start_id, start_id + nb_img_per_rank, 1)

            _inference_recursive_V2(
                img=img,
                n_h=n_h,
                id_list=id_list,
                pb_path=paths['optimized_pb_path'],
                conserve_nodes=conserve_nodes,
                hyper=hyper,
                comm=communicator
            )

        # save recon
        if rank == 0:
            reconstructor.reconstruct()
            recon = reconstructor.get_reconstruction()
            Image.fromarray(recon).save(paths['out_dir'] + 'step{}_{}.tif'.format(paths['step'], i))
            pbar1.update(1)


def _inference_recursive_V2(img=None, id_list=None, n_h=None, pb_path=None, conserve_nodes=None, hyper=None, comm=None):
    # load graph
    tf.reset_default_graph()
    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        _ = tf.import_graph_def(graph_def_optimized, return_elements=[conserve_nodes[-1]])
        G = tf.get_default_graph()
        X = G.get_tensor_by_name('import/' + 'new_input:0')
        y = G.get_tensor_by_name('import/' + conserve_nodes[-1] + ':0')
        bn = G.get_tensor_by_name('import/' + 'new_BN:0')
        do = G.get_tensor_by_name('import/' + 'new_dropout:0')

        batch_id = id_list.size // hyper['batch_size']
        batch_rest = id_list.size % hyper['batch_size']
        print(comm.Get_rank(), ': bid:{}, brst:{}'.format(batch_id, batch_rest))

        if batch_id != 0:
            for bid in range(batch_id):
                batch = []
                for id in id_list[bid * hyper['batch_size']: (bid + 1) * hyper['batch_size']]:
                    b = id // n_h
                    a = id % n_h
                    # concat patch to batch
                    batch.append(img[
                                 a * hyper['stride']: a * hyper['stride'] + hyper['patch_size'],
                                 b * hyper['stride']: b * hyper['stride'] + hyper['patch_size']
                                 ])

                batch = np.stack(batch, axis=0)
                # inference
                # batch = _minmaxscalar(batch)  # note: don't forget the minmaxscalar, since during training we put it
                batch = np.expand_dims(batch, axis=3)  # ==> (8, 512, 512, 1)
                feed_dict = {
                    X: batch,
                    do: 1.0,
                    bn: False,
                }

                if hyper['mode'] == 'classification':
                    output = sess.run(y, feed_dict=feed_dict)
                    output = customized_softmax_np(output)
                    comm.send([comm.Get_rank(), id_list[bid * hyper['batch_size']], output], dest=0, tag=tag_compute)
                else:
                    raise NotImplementedError('!')

        if batch_rest != 0:
            batch = []
            for id in id_list[-batch_rest:]:
                a = id % n_h
                b = id // n_h
                # concat patch to batch
                batch.append(img[
                             a * hyper['stride']: a * hyper['stride'] + hyper['patch_size'],
                             b * hyper['stride']: b * hyper['stride'] + hyper['patch_size']
                             ])

            batch = np.stack(batch, axis=0)
            # 1-padding for the last batch
            # note: 0-padding will give a wrong output shape
            batch = np.concatenate([batch, np.ones(
                (hyper['batch_size'] - batch_rest, *batch.shape[1:])
            )], axis=0)
            # inference
            # batch = _minmaxscalar(batch)  # note: don't forget the minmaxscalar, since during training we put it
            batch = np.expand_dims(batch, axis=3)  # ==> (8, 512, 512, 1)
            feed_dict = {
                X: batch,
                do: 1.0,
                bn: False,
            }

            if hyper['mode'] == 'classification':
                output = sess.run(y, feed_dict=feed_dict)
                output = customized_softmax_np(output)
                comm.send([comm.Get_rank(), id_list[-batch_rest], output[:batch_rest]], dest=0, tag=tag_compute)

            else:
                raise NotImplementedError('!')


if __name__ == '__main__':
    # graph_def_dir = './logs/2020_2_11_bs8_ps512_lrprogrammed_cs3_nc32_do0.1_act_leaky_aug_True_BN_True_mdl_LRCS_mode_classification_comment_DSC_rampdecay0.0001_k0.3_p1_wrapperWithoutMinmaxscaler_augWith_test_aug_GreyVar/hour10/'
    ckpt_path = args.ckpt_path
    save_pb_dir = '/'.join(ckpt_path.split('/')[:-2]) + '/pb/'

    c_nodes = [
            '{}/decoder/logits/identity'.format(re.search('mdl_(.*)_', ckpt_path)),
        ]

    # segment raw img per raw img

    paths = {
        'step': args.step,
        'working_dir': '/'.join(ckpt_path.split('/')[:-2]) + '/',
        'ckpt_path': args.ckpt_path,
        'save_pb_dir': save_pb_dir,
        'save_pb_path': save_pb_dir + 'frozen_step{}.pb'.format(args.step),
        'optimized_pb_path': save_pb_dir + 'optimized_step{}.pb'.format(args.step),
        'GPU': 0,
        'inference_dir': args.pred_path,
    }

    hyperparams = {
        'patch_size': args.window_size,
        'batch_size': args.batch_size,
        'nb_batch': None,
        'nb_patch': None,
        'stride': 10,
        'batch_normalization': True,
        'device_option': 'cpu',  #'cpu',
        'mode': 'classification',
        'nb_classes': args.n,
    }

    l_img_path = ['./testdata/0.tif']
    l_out = inference_recursive_V2(l_input_path=l_img_path, conserve_nodes=c_nodes, paths=paths, hyper=hyperparams)

