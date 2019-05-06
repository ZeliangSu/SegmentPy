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
    '''input: (tf.Graph())'''
    # fixme: enlarge to GraphDef
    for i in graph.get_operations():
        if len(i.outputs) is not 0:  #eliminate nodes like 'initializer' without tensoroutput
            for j in i.outputs:
                print('{}: {}'.format(i.name, j.get_shape().as_list()))

            # for j in i.outputs:
            #     print(j.get_shape())

