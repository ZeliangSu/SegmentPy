import tensorflow as tf
from tensorflow.core.framework import graph_pb2


def print_nodes_name(graph):
    '''input: (tf.Graph() or tf.GraphDef())'''
    if isinstance(graph, graph_pb2.GraphDef):
        for n in graph.node:
            print(n.name)
    else:
        for n in graph.as_graph_def().node:
            print(n.name)


def print_nodes_name_shape(graph):
    '''input: (tf.Graph()) or tf.GraphDef())'''
    # fixme: enlarge to GraphDef
    if isinstance(graph, graph_pb2.GraphDef):
        # convert GraphDef to Graph
        graph = tf.import_graph_def(graph)

    for i in graph.get_operations():
        if len(i.outputs) is not 0:  #eliminate nodes like 'initializer' without tensor output
            for j in i.outputs:
                print('{}: {}'.format(i.name, j.get_shape()))


def get_all_trainable_variables(metagraph_path):
    restorer = tf.train.import_meta_graph(
        metagraph_path + '.meta',
        clear_devices=True
    )

    with tf.Session() as sess:
        restorer.restore(sess, metagraph_path)
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        wn = [v.name for v in all_vars if v.name.endswith('w:0') and not v.name.startswith('dnn')]
        bn = [v.name for v in all_vars if v.name.endswith('b:0') and not v.name.startswith('dnn')]
        dnn_wn = [v.name for v in all_vars if v.name.endswith('w:0') and v.name.startswith('dnn')]
        dnn_bn = [v.name for v in all_vars if v.name.endswith('b:0') and v.name.startswith('dnn')]
        ws = [sess.run(v) for v in all_vars if v.name.endswith('w:0') and not v.name.startswith('dnn')]
        bs = [sess.run(v) for v in all_vars if v.name.endswith('b:0') and not v.name.startswith('dnn')]
        dnn_ws = [sess.run(v) for v in all_vars if v.name.endswith('b:0') and v.name.startswith('dnn')]
        dnn_bs = [sess.run(v) for v in all_vars if v.name.endswith('b:0') and v.name.startswith('dnn')]

    return wn, bn, ws, bs, dnn_wn, dnn_bn, dnn_ws, dnn_bs

