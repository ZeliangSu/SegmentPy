import tensorflow as tf
import numpy as np
from util import print_nodes_name, print_nodes_name_shape
from tensorflow.python.tools import inspect_checkpoint as chkpt
from tensorflow.python.tools import freeze_graph
# from tensorflow.python.tools import optimize_for_inference_lib

# path to checkpoint
ckpt = './dummy/ckpt/test.meta'

# visualize some tensors saved
chkpt.print_tensors_in_checkpoint_file('./dummy/ckpt/test', tensor_name='conv1/W', all_tensors=True)
chkpt.print_tensors_in_checkpoint_file('./dummy/ckpt/test', tensor_name='conv1/b', all_tensors=True)

# import saved meta graph
restorer = tf.train.import_meta_graph(ckpt)
graph = tf.get_default_graph()
print('\n\t\t\t**********before extract**********')
print_nodes_name(graph)
# print('\n\t\t\t**********Node names and shapes**********')
# print_nodes_name_shape(graph)

# decide which node to conserve while pruning
nodes_to_conserve = []
for n in graph.get_operations():
    if n.name.startswith('model/conv1'):
        nodes_to_conserve.append(n.name)

# extract subgraph
subgraph = tf.graph_util.extract_sub_graph(graph.as_graph_def(add_shapes=True), nodes_to_conserve)
# subgraph = tf.graph_util.remove_training_nodes(graph, protected_nodes=['model/conv1/Relu'])

print('\n\t\t\t**********after extract**********')
print_nodes_name(subgraph)
# print('\n\t\t\t**********Node names and shapes**********')
# print_nodes_name_shape(subgraph)
tf.reset_default_graph()

# cut the input pipeline branch
new_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='new_ph')
new_graph = tf.graph_util.import_graph_def(subgraph, input_map={'input/IteratorGetNext': new_ph},
                                       name='')  # '' removes the 'import/' prefix


freeze_graph.freeze_graph(input_graph='tensorflowModel.pbtxt',
                          input_saver="",
                          input_binary=False,
                          input_checkpoint='./dummy/ckpt/test.ckpt',
                          output_node_names="output/softmax",
                          restore_op_name="save/restore_all",  #useless argument with new version of this function
                          filename_tensor_name='',  #useless argument new version of this function
                          output_graph='./dummy/test.pb',
                          clear_devices=True,
                          initializer_nodes=""
                         )

# start loading graph and infering
with tf.Session(graph=new_graph) as sess:
    tf.summary.FileWriter('./dummy/tensorboard', sess.graph)
    restorer.restore(sess, './dummy/ckpt/test')
    print('\n\t\t\t**********restored weights**********')
    print_nodes_name_shape(tf.get_default_graph())
    subgraph = tf.graph_util.convert_variables_to_constants(sess, subgraph, output_node_names=['model/conv1/Relu'])
    # print('\n\t\t\t**********Node names and shapes**********')
    # print_nodes_name_shape(graph)
    sess.run('model/conv1/Relu', feed_dict={new_ph: np.zeros((1, 28, 28, 1))})