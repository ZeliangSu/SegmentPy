import tensorflow as tf
import numpy as np
from util import print_nodes_name, print_nodes_name_shape
ckpt = './dummy/ckpt/test.meta'


restorer = tf.train.import_meta_graph(ckpt)
graph = tf.get_default_graph()
# print('\n\t\t\t**********Node names**********')
# print_nodes_name(graph)
# print('\n\t\t\t**********Node names and shapes**********')
# print_nodes_name_shape(graph)


nodes_to_conserve = []
for n in graph.get_operations():
    if n.name.startswith('model') or n.name.startswith('conv1'):
        nodes_to_conserve.append(n.name)

# extract subgraph
subgraph = tf.graph_util.extract_sub_graph(graph.as_graph_def(add_shapes=True), nodes_to_conserve)
# subgraph = tf.graph_util.remove_training_nodes(subgraph, protected_nodes=['model/conv2/Conv2D', 'model/conv2/Relu'])


# print('\n\t\t\t**********Node names**********')
# print_nodes_name(subgraph)
# print('\n\t\t\t**********Node names and shapes**********')
# print_nodes_name_shape(subgraph)
tf.reset_default_graph()

new_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='new_ph')
graph = tf.graph_util.import_graph_def(subgraph, input_map={'model/conv1/Conv2D': new_ph})

with tf.Session(graph=graph) as sess:
    # restorer.restore(sess, './dummy/ckpt/test')
    print('\n\t\t\t**********Node names**********')
    print_nodes_name(tf.get_default_graph())
    # print('\n\t\t\t**********Node names and shapes**********')
    # print_nodes_name_shape(graph)
    sess.run('import/model/conv1/Relu', feed_dict={new_ph: np.zeros((1, 28, 28, 1))})