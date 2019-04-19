import tensorflow as tf
from util import print_nodes_name, print_nodes_name_shape
ckpt = './dummy/ckpt/test.meta'


restorer = tf.train.import_meta_graph(ckpt)

graph = tf.get_default_graph()
print('\n\t\t\t**********Node names**********')
print_nodes_name(graph)
# print('\n\t\t\t**********Node names and shapes**********')
# print_nodes_name_shape(graph)

nodes_to_conserve = []
for n in graph.get_operations():
    if n.name.startswith('model') and n.name.endswith('Relu'):
        nodes_to_conserve.append(n.name)

# extract subgraph
subgraph = tf.graph_util.extract_sub_graph(graph.as_graph_def(add_shapes=True), nodes_to_conserve)

print('\n\t\t\t**********Node names**********')
print_nodes_name(graph)
# print('\n\t\t\t**********Node names and shapes**********')
# print_nodes_name_shape(graph)

with tf.Session(graph=tf.graph_util.import_graph_def(subgraph)) as sess:
    # restorer.restore(sess, './dummy/ckpt/test')
    print('\n\t\t\t**********Node names**********')
    print_nodes_name(graph)
    # print('\n\t\t\t**********Node names and shapes**********')
    # print_nodes_name_shape(graph)