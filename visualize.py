import tensorflow as tf
import numpy as np
from util import print_nodes_name

# load ckpt
new_ph = tf.placeholder(tf.float32, name='new_ph')
restorer = tf.train.import_meta_graph(
    './dummy/ckpt/step5/ckpt.meta',
    input_map={'input_pipeline/input_cond/Merge_1': new_ph},
    clear_devices=True
)

input_graph = tf.get_default_graph()
input_graph_def = input_graph.as_graph_def()

# freeze to pb
with tf.Session() as sess:
    restorer.restore(sess, './dummy/ckpt/step5/ckpt')
    # print_nodes_name(sess.graph)
    tf.summary.FileWriter('./dummy/tensorboard/before_cut', sess.graph)
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
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=input_graph_def,
        output_node_names=conserve_nodes,
    )
    print_nodes_name(sess.graph)

    # write pb
    with tf.gfile.GFile('./dummy/pb/test.pb', 'wb') as f:  #'wb' stands for write binary
        f.write(output_graph_def.SerializeToString())

# use pb
path = './dummy/pb/test.pb'
with tf.gfile.GFile(path, mode='rb') as f:
    # init GraphDef()
    restored_graph_def = tf.GraphDef()
    # parse saved .pb to GraphDef()
    restored_graph_def.ParseFromString(f.read())


with tf.Graph().as_default() as graph:
    # import graph def
    tf.import_graph_def(
        graph_def=restored_graph_def,
        return_elements=conserve_nodes,
        name=''
    )

    # print_nodes_name(graph)
    # feed graph for inference
    new_input = graph.get_tensor_by_name('new_ph:0')
    ops = [graph.get_tensor_by_name(op_name + ':0') for op_name in conserve_nodes]

    with tf.Session(graph=graph) as sess:
        tf.summary.FileWriter('./dummy/tensorboard/after_cut', sess.graph)
        graph = tf.get_default_graph()
        # print_nodes_name(graph)

        res = sess.run(ops, feed_dict={new_input: np.ones((1, 72, 72, 1))})
        print(res.shape)
# save inference