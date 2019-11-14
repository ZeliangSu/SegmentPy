import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import os


def print_nodes_name(graph):
    """
    input:
    -------
        graph: (tf.Graph() or tf.GraphDef()) graph in which prints only the nodes' name
    return:
    -------
        None
    """
    if isinstance(graph, graph_pb2.GraphDef):
        for n in graph.node:
            print(n.name)
    else:
        for n in graph.as_graph_def().node:
            print(n.name)


def print_nodes_name_shape(graph):
    """
    input:
    -------
        graph: (tf.Graph()) or tf.GraphDef()) graph in which prints the nodes' name and their shapes
    return:
    -------
        None
    """
    # fixme: enlarge to GraphDef
    if isinstance(graph, graph_pb2.GraphDef):
        # convert GraphDef to Graph
        graph = tf.import_graph_def(graph)

    for i in graph.get_operations():
        if len(i.outputs) is not 0:  #eliminate nodes like 'initializer' without tensor output
            for j in i.outputs:
                print('{}: {}'.format(i.name, j.get_shape()))


def get_all_trainable_variables(metagraph_path):
    """
    input:
    -------
        metagraph_path: (string) indicate the path to find the metagraph of ckpt
    return:
    -------
        wn: (list of string) list of names of weights for all convolution and deconvolution layers
        bn: (list of string) list of names of bias for all convolution and deconvolution layers
        ws: (list of np.ndarray) list of weight matrices for all convolution and deconvolution layers
        bs: (list of np.ndarray) list of bias matrices for all convolution and deconvolution layers
        dnn_wn: (list of string) list of names of weights for all fully connected layers
        dnn_bn: (list of string) list of names of bias for all fully connected layers
        dnn_ws: (list of np.ndarray) list of weight matrices for all fully connected layers
        dnn_bs:(list of np.ndarray) list of bias matrices for all fully connected layers
    """
    tf.reset_default_graph()
    restorer = tf.train.import_meta_graph(
        metagraph_path + '.meta',
        clear_devices=True
    )

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        restorer.restore(sess, metagraph_path)
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        wn = [v.name for v in all_vars if v.name.endswith('w:0') and not v.name.startswith('dnn')]
        bn = [v.name for v in all_vars if v.name.endswith('b:0') and not v.name.startswith('dnn')]
        dnn_wn = [v.name for v in all_vars if v.name.endswith('w:0') and v.name.startswith('dnn')]
        dnn_bn = [v.name for v in all_vars if v.name.endswith('b:0') and v.name.startswith('dnn')]
        ws = [sess.run(v) for v in all_vars if v.name.endswith('w:0') and not v.name.startswith('dnn')]
        bs = [sess.run(v) for v in all_vars if v.name.endswith('b:0') and not v.name.startswith('dnn')]
        dnn_ws = [sess.run(v) for v in all_vars if v.name.endswith('w:0') and v.name.startswith('dnn')]
        dnn_bs = [sess.run(v) for v in all_vars if v.name.endswith('b:0') and v.name.startswith('dnn')]

    return wn, bn, ws, bs, dnn_wn, dnn_bn, dnn_ws, dnn_bs


def check_N_mkdir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir, exist_ok=True)


