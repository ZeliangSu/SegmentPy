import tensorflow as tf
import numpy as np
from util import print_nodes_name_shape, print_nodes_name
from writer import _resultWriter
import h5py as h5
import os

if os.name == 'posix':  #to fix MAC openMP bug
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def convert_ckpt2pb(input=None, ckpt_path='./dummy/ckpt/step5/ckpt', pb_path='./dummy/pb/test.pb', conserve_nodes=None):
    restorer = tf.train.import_meta_graph(
        ckpt_path + '.meta',
        input_map={
            'input_pipeline/input_cond/Merge_1': input
        },
        clear_devices=True
    )

    input_graph = tf.get_default_graph()
    input_graph_def = input_graph.as_graph_def()

    # freeze to pb
    with tf.Session() as sess:
        restorer.restore(sess, ckpt_path)
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


# build different block
def built_diff_block(patch_size=72):
    # diff node new graph
    with tf.Graph().as_default() as g_diff:
        with tf.name_scope('diff_block'):
            # two phs for passing values
            label_ph = tf.placeholder(tf.int32, shape=[None, patch_size, patch_size, 1], name='label_ph')
            res_ph = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 1], name='res_ph')
            # diff op
            diff = tf.cast(
                tf.not_equal(
                    label_ph,
                    tf.reshape(
                        tf.cast(
                            res_ph,
                            tf.int32
                        ),
                        [-1, 72, 72, 1]
                    )
                ),
                tf.int32,
                name='diff_img'
            )  #todo: doing like this isn't optimal

        g_diff_def = g_diff.as_graph_def()
    return g_diff_def


def join_diff_to_mainGraph(g_diff_def, conserve_nodes, path='./dummy/pb/test.pb'):
    # load main graph pb
    with tf.gfile.GFile(path, mode='rb') as f:
        # init GraphDef()
        restored_graph_def = tf.GraphDef()
        # parse saved .pb to GraphDef()
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as g_combined:
        # import graph def
        pred = tf.import_graph_def(
            graph_def=restored_graph_def,
            return_elements=['model/decoder/deconv8bisbis/relu:0'],
            name=''  #note: '' so that won't have import/ prefix
        )

        # join diff def
        tf.import_graph_def(
            g_diff_def,
            input_map={'diff_block/res_ph:0': tf.convert_to_tensor(pred)},
            return_elements=['diff_block/diff_img'],
            name=''
        )

        # prepare feed_dict for inference
        ops_dict = {
            'ops': [g_combined.get_tensor_by_name(op_name + ':0') for op_name in conserve_nodes],
            'diff_img': g_combined.get_tensor_by_name('diff_block/diff_img:0'),
        }

        return g_combined, ops_dict


def load_mainGraph(conserve_nodes, path='./dummy/pb/test.pb'):
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
            return_elements=['model/decoder/deconv8bisbis/relu:0'],
            name=''  # note: '' so that won't have import/ prefix
        )

    # prepare feed_dict for inference
    ops_dict = {
        'ops': [g_main.get_tensor_by_name(op_name + ':0') for op_name in conserve_nodes],
        }
    return g_main, ops_dict


def run_nodes_and_save_partial_res(g_combined, ops_dict, conserve_nodes):
    with g_combined.as_default() as g_combined:
        new_input = g_combined.get_tensor_by_name('new_ph:0')
        dropout_input = g_combined.get_tensor_by_name('input_pipeline/dropout_prob:0')
        new_label = g_combined.get_tensor_by_name('diff_block/label_ph:0')
        # run inference
        with tf.Session(graph=g_combined) as sess:
            # tf.summary.FileWriter('./dummy/tensorboard/after_combine', sess.graph)
            print_nodes_name_shape(sess.graph)

            # write firstly input and output images
            imgs = [h5.File('./proc/test/72/{}.h5'.format(i))['X'] for i in range(200)]
            _resultWriter(imgs, 'input')
            label = [h5.File('./proc/test/72/{}.h5'.format(i))['y'] for i in range(200)]
            _resultWriter(label, 'label')

            feed_dict = {
                new_input: np.array(imgs).reshape(200, 72, 72, 1),
                dropout_input: 1.0,
                new_label: np.array(label).reshape(200, 72, 72, 1),
            }

            # run partial results operations and diff block
            res, res_diff = sess.run([ops_dict['ops'], ops_dict['diff_img']], feed_dict=feed_dict)

            # note: save partial/final inferences of the first image
            for layer_name, tensors in zip(conserve_nodes, res):
                if tensors.ndim == 4 or 2:
                    tensors = tensors[0]  # todo: should generalize to batch
                _resultWriter(tensors, layer_name=layer_name.split('/')[-2], path='./result/test/')  #for cnn outputs shape: [batch, w, h, nb_conv]

            # note: save diff of all imgs
            _resultWriter(np.transpose(np.squeeze(res_diff), (1, 2, 0)), 'diff')  #for diff output shape: [batch, w, h, 1]



