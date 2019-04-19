import tensorflow as tf

def print_nodes_name(graph):
    '''input: (tf.Graph())'''
    for n in graph.as_graph_def().node:
        print(n.name)

def print_nodes_name_shape(graph):
    '''input: (tf.Graph())'''
    for i in graph.get_operations():
        if len(i.outputs) is not 0:  #eliminate nodes like 'initializer' without tensoroutput
            for j in i.outputs:
                print('{}: {}'.format(i.name, j.get_shape()))

            # for j in i.outputs:
            #     print(j.get_shape())

