import tensorflow as tf
import numpy as np
from util import print_nodes_name_shape, print_nodes_name, check_N_mkdir
from writer import _resultWriter
import h5py as h5
import os
import warnings


if os.name == 'posix':  #to fix MAC openMP bug
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def convert_ckpt2pb(input=None, paths=None, conserve_nodes=None):
    """
    inputs:
    -------
        input: (input pipeline of tf.Graph())
        ckpt_path: (str)
        pb_path: (str)
        conserve_nodes: (list of string)

    return:
    -------
        None
    """
    check_N_mkdir(paths['save_pb_dir'])
    pb_path = paths['save_pb_path']

    if not os.path.exists(pb_path):
        restorer = tf.train.import_meta_graph(
            paths['ckpt_path'] + '.meta',
            input_map={
                'input_pipeline/input_cond/Merge_1': input,
            },
            clear_devices=True
        )

        input_graph = tf.get_default_graph()
        input_graph_def = input_graph.as_graph_def()

        # freeze to pb
        with tf.Session() as sess:
            restorer.restore(sess, paths['ckpt_path'])
            # print_nodes_name_shape(sess.graph)
            # tf.summary.FileWriter('./dummy/tensorboard/before_cut', sess.graph)
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess=sess,  #note: variables are always in a session
                input_graph_def=input_graph_def,
                output_node_names=conserve_nodes,
            )
            # print_nodes_name_shape(sess.graph)

            # write pb
            with tf.gfile.GFile(pb_path, 'wb') as f:  #'wb' stands for write binary
                f.write(output_graph_def.SerializeToString())
    else:
        warnings.warn('pb file exists already!')
        pass


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


def inference_and_save_partial_res(g_main, ops_dict, conserve_nodes, input_dir=None, rlt_dir=None):
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
    with g_main.as_default() as g_main:
        new_input = g_main.get_tensor_by_name('new_ph:0')
        dropout_input = g_main.get_tensor_by_name('dropout_prob:0')

        # run inference
        with tf.Session(graph=g_main) as sess:
            # tf.summary.FileWriter('./dummy/tensorboard/after_combine', sess.graph)
            print_nodes_name_shape(sess.graph)

            # write firstly input and output images
            imgs = [h5.File(input_dir + '{}.h5'.format(i))['X'] for i in range(100)]
            _resultWriter(imgs, 'input', path=rlt_dir)
            label = [h5.File(input_dir + '{}.h5'.format(i))['y'] for i in range(100)]
            _resultWriter(label, 'label', path=rlt_dir)
            img_size = np.array(imgs[0]).shape[1]

            feed_dict = {
                new_input: np.array(imgs).reshape((100, img_size, img_size, 1)),
                dropout_input: 1.0,
            }

            # run partial results operations and diff block
            res = sess.run(ops_dict['ops'], feed_dict=feed_dict)

            # note: save partial/final inferences of the first image
            for layer_name, tensors in zip(conserve_nodes, res):
                try:
                    if tensors.ndim == 4 or 2:
                        if layer_name.split('/')[-2] != 'logits':
                            tensors = tensors[0]  # todo: should generalize to batch
                        else:
                            _tensors = [np.squeeze(tensors[i]) for i in range(tensors.shape[0])]
                            tensors = _tensors
                except:
                    pass
                _resultWriter(tensors, layer_name=layer_name.split('/')[-2],
                              path=rlt_dir)  # for cnn outputs shape: [batch, w, h, nb_conv]

    # calculate diff by numpy
    res_diff = np.equal(np.asarray(np.squeeze(res[-1]), dtype=np.int), np.asarray(label))
    res_diff = np.asarray(res_diff, dtype=np.int)

    # note: save diff of all imgs
    _resultWriter(np.transpose(res_diff, (1, 2, 0)), 'diff',
                  path=rlt_dir)  # for diff output shape: [batch, w, h, 1]
