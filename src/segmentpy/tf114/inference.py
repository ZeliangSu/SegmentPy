import tensorflow as tf
import numpy as np
import os
from segmentpy.tf114.util import check_N_mkdir, read_pb
from itertools import product
from PIL import Image
from tqdm import tqdm
from segmentpy.tf114.input import _minmaxscalar
from segmentpy.tf114.util import print_nodes_name, get_list_fnames, load_img, dimension_regulator
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
from segmentpy.tf114.layers import customized_softmax_np
import argparse
import re

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)

tag_compute = 1002

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #todo: uncomment this line will infect the detection of GPU in SegmentPy

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

            # uncomment following for checking the softmax
            # for k in range(3):
            #     Image.fromarray(batch[i][:, :, k]).save('./dummy/{}.tif'.format(k))

            tmp = np.argmax(batch[i], axis=2)  # (H, W)
            out = []
            for j in range(self.nb_cls):
                onehot = np.zeros(tmp.shape)
                onehot[np.where(tmp == j)] = 1
                out.append(onehot)
            out = np.stack(out, axis=2)  # (H, W, C)
            self.result[a * self.stride: a * self.stride + self.p_h, b * self.stride: b * self.stride + self.p_w, :] += out

    def reconstruct(self):
        print('***Reconostructing')

        # uncomment following for checking the softmax
        # for i in range(self.result.shape[2]):
        #     Image.fromarray(self.result[:, :, i]).save('./predict/result/{}.tif'.format(i))

        self.result = np.argmax(self.result, axis=2).astype(float)

    def get_reconstruction(self):
        return self.result

    def get_nb_patch(self):
        return self.n_h, self.n_w


class reconstructor_V3_cls():
    '''can be used by models which has resizable input (without dense connected layer/pur convolutional net)'''
    def __init__(self, image_size, z_len, nb_class, maxp_times=3):
        self.img_size = image_size
        self.z_len = z_len
        self.nb_cls = nb_class
        down = pow(2, maxp_times)
        a, b = self.img_size[0] // down, self.img_size[1] // down
        self.result = np.zeros((self.z_len, a * down, b * down), dtype=np.int8)

    def write_slice(self,
                    nn_output,
                    slice_id
                    ):
        seg = np.argmax(np.squeeze(nn_output), axis=2).astype(np.int8)
        self.result[slice_id, :, :] = seg

    def get_volume(self):
        return self.result.astype(np.float32)  # convert to float for saving in .tif


def freeze_ckpt_for_inference(paths=None, hyper=None, conserve_nodes=None):
    assert isinstance(paths, dict), 'The paths parameter expected a dictionnay but other type is provided'
    assert isinstance(hyper, dict), 'The hyper parameter expected a dictionnay but other type is provided'
    assert isinstance(conserve_nodes, list), 'The name of the conserve node should be in a list'

    # clean graph first
    tf.reset_default_graph()

    # freeze ckpt then convert to pb
    new_input = tf.placeholder(tf.float32, shape=[None, None, None, 10 if hyper['feature_map'] else 1], name='new_input')  # note: resize the input while inferencing
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
    try:
        optimize_graph_def = optimize_for_inference(
            input_graph_def=graph_def,
            input_node_names=['new_input', 'new_BN', 'new_dropout'],
            output_node_names=conserve_nodes,
            placeholder_type_enum=[dtypes.float32.as_datatype_enum,
                                   dtypes.bool.as_datatype_enum,
                                   dtypes.float32.as_datatype_enum,
                                   ]
        )
    except:
        optimize_graph_def = optimize_for_inference(
            input_graph_def=graph_def,
            input_node_names=['new_input', 'new_BN'],
            output_node_names=conserve_nodes,
            placeholder_type_enum=[dtypes.float32.as_datatype_enum,
                                   dtypes.bool.as_datatype_enum,
                                   dtypes.float32.as_datatype_enum,
                                   ]
        )
    with tf.gfile.GFile(paths['optimized_pb_path'], 'wb') as f:
        f.write(optimize_graph_def.SerializeToString())


def inference_recursive_V2(l_input_path=None, conserve_nodes=None, paths=None, hyper=None, normalization=1e-3):
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
        img = np.asarray(Image.open(path)) * normalization
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
                comm=communicator,
            )

        # save recon
        if rank == 0:
            reconstructor.reconstruct()
            recon = reconstructor.get_reconstruction()
            Image.fromarray(recon).save(paths['out_dir'] + 'step{}_{}.tif'.format(paths['step'], i))
            pbar1.update(1)


def _inference_recursive_V2(img=None, id_list=None, n_h=None, pb_path=None, conserve_nodes=None, hyper=None, comm=None):
    # load graph
    graph_def_optimized = read_pb(pb_path)

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        _ = tf.import_graph_def(graph_def_optimized, return_elements=[conserve_nodes[-1]])
        G = tf.get_default_graph()
        X = G.get_tensor_by_name('import/' + 'new_input:0')
        y = G.get_tensor_by_name('import/' + conserve_nodes[-1] + ':0')
        bn = G.get_tensor_by_name('import/' + 'new_BN:0')
        try:
            do = G.get_tensor_by_name('import/' + 'new_dropout:0')
        except Exception as e:
            print(e)
            print('drop out not exists in graph')
            do = None
            pass

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
                                 a * hyper['stride']: a * hyper['stride'] + hyper['patch_size'],  # fixme
                                 b * hyper['stride']: b * hyper['stride'] + hyper['patch_size']   # fixme
                                 ])

                batch = np.stack(batch, axis=0)
                # inference
                # batch = _minmaxscalar(batch)  # note: don't forget the minmaxscalar, since during training we put it
                batch = np.expand_dims(batch, axis=3)  # ==> (8, 512, 512, 1)

                if do is not None:
                    feed_dict = {
                        X: batch,
                        do: 1.0,
                        bn: False,
                    }
                else:
                    feed_dict = {
                        X: batch,
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

            if do is not None:
                feed_dict = {
                    X: batch,
                    do: 1.0,
                    bn: False,
                }
            else:
                feed_dict = {
                    X: batch,
                    bn: False,
                }

            if hyper['mode'] == 'classification':
                output = sess.run(y, feed_dict=feed_dict)
                output = customized_softmax_np(output)
                comm.send([comm.Get_rank(), id_list[-batch_rest], output[:batch_rest]], dest=0, tag=tag_compute)

            else:
                raise NotImplementedError('!')


def inference_recursive_V3(l_input_path=None, conserve_nodes=None, paths=None, hyper=None, norm=1e-3):
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

        for img in l_img_path:
            logger.debug(img)

        check_N_mkdir(paths['inference_dir'])
        freeze_ckpt_for_inference(paths=paths, hyper=hyper,
                                  conserve_nodes=conserve_nodes)
        optimize_pb_for_inference(paths=paths,
                                  conserve_nodes=conserve_nodes)
        reconstructor = reconstructor_V3_cls(
            image_size=load_img(l_input_path[0]).shape,
            z_len=len(l_input_path),
            nb_class=hyper['nb_classes'],
            maxp_times=hyperparams['maxp_times']
        )
        pbar1 = tqdm(total=len(l_input_path))

    # ************************************************************************************************ I'm a Barrier
    communicator.Barrier()

    # reconstruct volumn
    remaining = len(l_input_path)
    nb_img_per_rank = remaining // (nb_process - 1)
    rest_img = remaining % (nb_process - 1)
    print(nb_img_per_rank, rest_img, nb_process)

    if rank == 0:
        # start gathering batches from other rank
        s = MPI.Status()
        communicator.Probe(status=s)
        while remaining > 0:
            if s.tag == tag_compute:
                # receive outputs
                slice_id, out_batch = communicator.recv(tag=tag_compute)
                logger.debug(slice_id)
                reconstructor.write_slice(out_batch, slice_id)

                # progress
                remaining -= 1
                pbar1.update(1)

    else:
        if (rank - 1) < rest_img:
            start_id = (rank - 1) * (nb_img_per_rank + 1)
            id_list = np.arange(start_id, start_id + nb_img_per_rank + 1, 1)
        else:
            start_id = (rank - 1) * nb_img_per_rank + rest_img
            id_list = np.arange(start_id, start_id + nb_img_per_rank, 1)

        logger.debug('{}: {}'.format(rank, id_list))
        _inference_recursive_V3(
            l_input_path=l_input_path,
            id_list=id_list,
            pb_path=paths['optimized_pb_path'],
            conserve_nodes=conserve_nodes,
            hyper=hyper,
            comm=communicator,
            maxp_times=hyperparams['maxp_times'],
            normalization=norm
        )

    # ************************************************************************************************ I'm a Barrier
    communicator.Barrier()

    # save recon
    if rank == 0:
        recon = reconstructor.get_volume()
        for i in tqdm(range(len(l_input_path)), desc='writing data'):
            Image.fromarray(recon[i]).save(paths['inference_dir'] + 'step{}_{}.tif'.format(paths['step'], i))


def _inference_recursive_V3(l_input_path: list,
                            id_list: np.ndarray,
                            pb_path: str,
                            conserve_nodes: list,
                            hyper: dict,
                            comm=None,
                            normalization=1e-3,
                            maxp_times=3):
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
        try:
            do = G.get_tensor_by_name('import/' + 'new_dropout:0')
        except Exception as e:
            print(e)
            print('drop out not exists in graph')
            do = None
            pass

        print(comm.Get_rank(), ': nb of inference:{}'.format(len(id_list)))
        if id_list.size != 0:
            for id in np.nditer(id_list):
                # note: the following dimensions should be multiple of 8 if 3x Maxpooling
                logger.debug('rank {}: {}'.format(comm.Get_rank(), id))
                img = load_img(l_input_path[id]) / normalization
                img = dimension_regulator(img, maxp_times=maxp_times)

                batch = img.reshape((1, *img.shape, 1))
                # inference
                if do is not None:
                    feed_dict = {
                        X: batch,
                        do: 1.0,
                        bn: False,
                    }
                else:
                    feed_dict = {
                        X: batch,
                        bn: False,
                    }

                if hyper['mode'] == 'classification':
                    output = sess.run(y, feed_dict=feed_dict)
                    output = customized_softmax_np(output)
                    comm.send([id, output], dest=0, tag=tag_compute)
                else:
                    raise NotImplementedError('!')


if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser('main.py')
    parser.add_argument('-ckpt', '--ckpt_path', type=str, metavar='', required=True, help='.meta path')
    parser.add_argument('-raw', '--raw_dir', type=str, metavar='', required=True, help='raw tomograms folder path')
    parser.add_argument('-pred', '--pred_dir', type=str, metavar='', required=True,
                        help='where to put the segmentation')
    parser.add_argument('-corr', '--correction', type=float, metavar='', required=True, help='manually correct the input image by a coefficient')
    # todo:
    parser.add_argument('-cls', '--nb_cls', type=int, metavar='', required=False, default=3, help='nb of classes in the vol (automatic in the future version)')
    parser.add_argument('-bs', '--batch_size', type=int, metavar='', required=False, default=8,
                        help='identical as the trained model')
    parser.add_argument('-ws', '--window_size', type=int, metavar='', required=False, default=512,
                        help='identical as the trained model')
    args = parser.parse_args()
    logger.debug(args)

    # graph_def_dir = './logs/2020_2_11_bs8_ps512_lrprogrammed_cs3_nc32_do0.1_act_leaky_aug_True_BN_True_mdl_LRCS_mode_classification_comment_DSC_rampdecay0.0001_k0.3_p1_wrapperWithoutMinmaxscaler_augWith_test_aug_GreyVar/hour10/'
    ckpt_path = args.ckpt_path.replace('.meta', '')
    save_pb_dir = '/'.join(ckpt_path.split('/')[:-2]) + '/pb/'
    mdl_name = re.search('mdl_([A-Za-z]+\d*)', ckpt_path).group(1)

    c_nodes = [
        '{}/decontractor/logits/identity'.format(mdl_name) if 'Unet' in mdl_name else '{}/decoder/logits/identity'.format(mdl_name),
        ]

    # segment raw img per raw img
    step = re.search('step(\d+)', ckpt_path).group(1)
    paths = {
        'step': step,
        'working_dir': '/'.join(ckpt_path.split('/')[:-2]) + '/',
        'ckpt_path': ckpt_path,
        'save_pb_dir': save_pb_dir,
        'save_pb_path': save_pb_dir + 'frozen_step{}.pb'.format(step),
        'optimized_pb_path': save_pb_dir + 'optimized_step{}.pb'.format(step),
        'GPU': 0,
        'inference_dir': args.pred_dir,
        'raw_dir': args.raw_dir,
    }

    logger.debug(paths['ckpt_path'])

    hyperparams = {
        'patch_size': args.window_size,
        'batch_size': args.batch_size,
        'nb_batch': None,
        'nb_patch': None,
        'stride': 200,  # stride == 30, reconstuction time ~ 1min per tomogram
        'batch_normalization': True,
        'device_option': 'cpu',  #'cpu',
        'mode': 'classification',
        'nb_classes': args.nb_cls,
        'feature_map': True if mdl_name in ['LRCS8', 'LRCS9', 'LRCS10', 'Unet3'] else False,
        'maxp_times': 4 if mdl_name in ['Unet', 'Segnet', 'Unet5', 'Unet6'] else 3,
        'correction': args.correction,
    }

    l_img_path = [paths['raw_dir'] + f for f in sorted(os.listdir(paths['raw_dir']))]
    inference_recursive_V3(l_input_path=l_img_path,
                           conserve_nodes=c_nodes,
                           paths=paths,
                           hyper=hyperparams,
                           norm=args.correction)
    print('ha')
