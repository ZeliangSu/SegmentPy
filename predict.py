import tensorflow as tf
import numpy as np


def load_pb(path='./dummy/pb/test.pb', pred_node='model/decoder/deconv8bisbis/relu:0'):
    with tf.gfile.GFile(path, 'rb') as f:
        # init Graph_def
        restored_graph_def = tf.GraphDef()
        # load pb
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as g_pred:
        tf.import_graph_def(
            graph_def=restored_graph_def,
            return_elements=[pred_node],
            name=''
        )
    return g_pred


g_pred = load_pb()
with tf.Graph().as_default() as new_graph:
    new_input = new_graph.get_tensor_by_name('new_ph:0')
    dropout_input = new_graph.get_tensor_by_name('input_pipeline/dropout_prob:0')
    new_label = new_graph.get_tensor_by_name('diff_block/label_ph:0')



