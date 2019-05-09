import tensorflow as tf
import numpy as np
from util import print_nodes_name_shape, print_nodes_name
from writer import _tifsWriter
import h5py as h5
import os

if os.name == 'posix':  #to fix MAC openMP bug
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def convert_ckpt2pb(input=None, ckpt_path='./dummy/ckpt/step5/ckpt.meta', pb_path='./dummy/pb/test.pb', conserve_nodes=None):
    restorer = tf.train.import_meta_graph(
        ckpt_path,
        input_map={
            'input_pipeline/input_cond/Merge_1': input
        },
        clear_devices=True
    )

    input_graph = tf.get_default_graph()
    input_graph_def = input_graph.as_graph_def()

    # freeze to pb
    with tf.Session() as sess:
        restorer.restore(sess, './dummy/ckpt/step5/ckpt')
        # print_nodes_name_shape(sess.graph)
        tf.summary.FileWriter('./dummy/tensorboard/before_cut', sess.graph)
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess=sess,  #note: variables are always in a session
            input_graph_def=input_graph_def,
            output_node_names=conserve_nodes,
        )
        # print_nodes_name_shape(sess.graph)

        # write pb
        with tf.gfile.GFile(pb_path, 'wb') as f:  #'wb' stands for write binary
            f.write(output_graph_def.SerializeToString())

# load ckpt
new_ph = tf.placeholder(tf.float32, shape=[200, 72, 72, 1], name='new_ph')
bs_ph = tf.placeholder(tf.int32, shape=None, name='bs_ph')

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
            'model/dnn/dnn1/relu',
            'model/dnn/dnn2/relu',
            'model/dnn/dnn3/relu',
            'model/decoder/deconv5/relu',
            'model/decoder/deconv5bis/relu',
            'model/decoder/deconv6/relu',
            'model/decoder/deconv6bis/relu',
            'model/decoder/deconv7bis/relu',
            'model/decoder/deconv7bis/relu',
            'model/decoder/deconv8/relu',
            'model/decoder/deconv8bis/relu',
            'model/decoder/deconv8bisbis/relu',
        ]

# convert ckpt to pb
convert_ckpt2pb(input=new_ph, conserve_nodes=conserve_nodes)

# diff node new graph
with tf.Graph().as_default() as g_diff:
    with tf.name_scope('diff_block'):
        # two phs for passing values
        label_ph = tf.placeholder(tf.int32, shape=[None, 72, 72, 1], name='label_ph')
        res_ph = tf.placeholder(tf.float32, shape=[None, 72, 72, 1], name='res_ph')
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

# use pb
path = './dummy/pb/test.pb'
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
    diff_res = tf.import_graph_def(
        g_diff_def,
        input_map={'diff_block/res_ph:0': tf.convert_to_tensor(pred)},
        return_elements=['diff_block/diff_img'],
        name=''
    )

    # prepare feed_dict for inference
    new_input = g_combined.get_tensor_by_name('new_ph:0')
    dropout_input = g_combined.get_tensor_by_name('input_pipeline/dropout_prob:0')
    ops = [g_combined.get_tensor_by_name(op_name + ':0') for op_name in conserve_nodes]
    new_label = g_combined.get_tensor_by_name('diff_block/label_ph:0')
    diff_img = g_combined.get_tensor_by_name('diff_block/diff_img:0')

    # run inference
    with tf.Session(graph=g_combined) as sess:
        tf.summary.FileWriter('./dummy/tensorboard/after_combine', sess.graph)
        graph = tf.get_default_graph()
        print_nodes_name_shape(sess.graph)

        imgs = [h5.File('./proc/test/72/{}.h5'.format(i))['X'] for i in range(200)]
        _tifsWriter(imgs, 'input')
        label = [h5.File('./proc/test/72/{}.h5'.format(i))['y'] for i in range(200)]
        _tifsWriter(label, 'label')
        res, res_diff = sess.run([ops, diff_img], feed_dict={
            new_input: np.array(imgs).reshape(200, 72, 72, 1),
            dropout_input: 1.0,
            new_label: np.array(label).reshape(200, 72, 72, 1)
        })
        for elt in res:
            print(elt.shape)

        #note: save partial/final inferences of the first image
        for layer_name, tensors in zip(conserve_nodes, res):
            if tensors.ndim == 4 or 2:
                tensors = tensors[0]  #todo: should generalize to batch
            _tifsWriter(tensors, layer_name=layer_name.split('/')[-2], path='./result/test/')  #for cnn outputs shape: [batch, w, h, nb_conv]

        # save diff of all imgs
        _tifsWriter(np.transpose(np.squeeze(res_diff), (1, 2, 0)), 'diff')  #for diff output shape: [batch, w, h, 1]

