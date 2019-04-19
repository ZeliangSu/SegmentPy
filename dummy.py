import tensorflow as tf
from input import _pyfn_wrapper

ckpt = './logs/2019_4_19/hour13/ckpt/step5/ckpt.meta'
img = ['./proc/test/72/100.h5', './proc/test/72/200.h5']

X, y = _pyfn_wrapper(img[0], 72)

restorer = tf.train.import_meta_graph(ckpt)
                                      # input_map={'input_pipeline/input_cond/Merge_1:0': tf.convert_to_tensor(X)},
                                      # return_elements=['model/decoder/deconv8bisbis/deconv8bisbis_relu'])

graph = tf.get_default_graph()

# drop_ph = graph_def.get_tensor_by_name('dropout_prob:0')
# tr_init = graph_def.get_operation_by_name('input_pipelinetrain/iter_init_op')
# f_tr_ph = graph_def.get_tensor_by_name('input_pipelinetrain/fnames_ph:0')
# tr_ps_ph = graph_def.get_tensor_by_name('input_pipelinetrain/patch_size_ph:0')
# tt_init = graph_def.get_operation_by_name('input_pipelinetest/iter_init_op')
# f_tt_ph = graph_def.get_tensor_by_name('input_pipelinetest/fnames_ph:0')
# tt_ps_ph = graph_def.get_tensor_by_name('input_pipelinetest/patch_size_ph:0')

# tot_ph = graph_def.get_tensor_by_name('training_type:0')
# drop_ph = graph_def.get_tensor_by_name('dropout_prob:0')
# loss_op = graph.get_tensor_by_name('model/operation/loss_fn/loss:0')
# acc_op = graph.get_tensor_by_name('model/operation/accuracy/accuracy:0')
#
# conv1 = graph.get_tensor_by_name('model/encoder/conv1/relu:0')
# conv1bis = graph.get_tensor_by_name('model/encoder/conv1bis/relu:0')
# conv2 = graph.get_tensor_by_name('model/encoder/conv2/relu:0')
# conv2bis = graph.get_tensor_by_name('model/encoder/conv2bis/relu:0')
# conv3 = graph.get_tensor_by_name('model/encoder/conv3/relu:0')
# conv3bis = graph.get_tensor_by_name('model/encoder/conv3bis/relu:0')
# conv4 = graph.get_tensor_by_name('model/encoder/conv4/relu:0')
# conv4bis = graph.get_tensor_by_name('model/encoder/conv4bis/relu:0')
# conv4bisbis = graph.get_tensor_by_name('model/encoder/conv4bisbis/relu:0')
# deconv5 = graph.get_tensor_by_name('model/decoder/deconv5/relu:0')
# deconv5bis = graph.get_tensor_by_name('model/decoder/deconv5bis/relu:0')
# deconv6 = graph.get_tensor_by_name('model/decoder/deconv6/relu:0')
# deconv6bis = graph.get_tensor_by_name('model/decoder/deconv6bis/relu:0')
# deconv7 = graph.get_tensor_by_name('model/decoder/deconv7/relu:0')
# deconv7bis = graph.get_tensor_by_name('model/decoder/deconv7bis/relu:0')
# deconv8 = graph.get_tensor_by_name('model/decoder/deconv8/relu:0')
# deconv8bis = graph.get_tensor_by_name('model/decoder/deconv8bis/relu:0')
# deconv8bisbis = graph.get_tensor_by_name('model/decoder/deconv8bisbis/relu:0')

nodes_to_conserve = []
for n in graph.get_operations():
    if n.name.startswith('model') and n.name.endswith('relu'):
        nodes_to_conserve.append(n.name)

subgraph = tf.graph_util.extract_sub_graph(graph.as_graph_def(add_shapes=True), nodes_to_conserve)


with tf.Session(graph=tf.graph_util.import_graph_def(subgraph)) as sess:  #import_graph_def convert graphdef to tf.Graph
    restorer.restore(sess, './logs/2019_4_19/hour13/ckpt/step5/ckpt')
    for i in tf.get_default_graph().get_operations():
        print(i.name)
        for j in i.outputs:
            print('OUTPUTS:', j.get_shape())



    print('\n********** sub graph')
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)

    loss, acc, c1, c1b, c2, c2b, c3, c3b, c4, c4b, c4bb, d5, d5b, d6, d6b, d7, d7b, d8, d8b, d8bb = sess.run([conv1,
                                                                                                        conv1bis,
                                                                                                        conv2,
                                                                                                        conv2bis,
                                                                                                        conv3,
                                                                                                        conv3bis,
                                                                                                        conv4,
                                                                                                        conv4bis,
                                                                                                        conv4bisbis,
                                                                                                        deconv5,
                                                                                                        deconv5bis,
                                                                                                        deconv6,
                                                                                                        deconv6bis,
                                                                                                        deconv7,
                                                                                                        deconv7bis,
                                                                                                        deconv8,
                                                                                                        deconv8bis,
                                                                                                        deconv8bisbis,
                                                                                                        loss_op,
                                                                                                        acc_op],
                                                                                                             # feed_dict={drop_ph: 1.0}
                                                                                                             )

    print(loss, acc)
    print(c1.shape, c1b.shape)