import tensorflow as tf


def init_weights(shape, name='weights'):
    with tf.variable_scope(name):
        return tf.get_variable('w', shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def init_bias(shape, name='bias'):
    with tf.variable_scope(name):
        return tf.get_variable('b', shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def max_pool_2by2(x, name=''):
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')


def up_2by2(input_layer, name=''):
    with tf.name_scope(name):
        return tf.image.resize_nearest_neighbor(input_layer, [2 * input_layer.shape[1], 2 * input_layer.shape[2]], name='up')


def placeholder(tensor_shape, name=''):
    with tf.name_scope(name):
        tensor_ph = tf.placeholder(tf.float32, shape=tensor_shape, name='ph')
        return tensor_ph, tf.shape(tensor_ph)[0]


def conv2d_layer(input_layer, shape, name='', stride=1):
    with tf.name_scope(name):
        W = init_weights(shape, name)  # [conv_height, conv_width, in_channels, output_channels]
        b = init_bias([shape[3]], name)
        output = tf.nn.conv2d(input_layer, W, strides=[1, stride, stride, 1], padding='SAME', name='conv') + b
        output_activation = tf.nn.relu(output, name='relu')
        return output_activation, tf.summary.merge([tf.summary.histogram("weights", W),
                                                   tf.summary.histogram("bias", b),
                                                   tf.summary.histogram("layer", output),
                                                   tf.summary.histogram("activations", output_activation)
                                                   ])

def conv2d_transpose_layer(input_layer, shape, dyn_batch_size, stride=1, name=''):
    with tf.name_scope(name):
        shape = [shape[0], shape[1], shape[3], shape[2]]  # switch in/output channels [height, width, output_channels, in_channels]
        W = init_weights(shape, name)
        b = init_bias([shape[2]], name)
        transpose = tf.nn.conv2d_transpose(input_layer, W, output_shape=(dyn_batch_size,
                                                                        int(input_layer.shape[1]),
                                                                        int(input_layer.shape[2]),
                                                                        int(W.shape[2])),
                                           strides=[1, stride, stride, 1], padding='SAME', name='deconv')
        output = transpose + b
        output_activation = tf.nn.relu(output, name='relu')
        return output_activation, tf.summary.merge([tf.summary.histogram("weights", W),
                                                   tf.summary.histogram("bias", b),
                                                   tf.summary.histogram("layer", output),
                                                   tf.summary.histogram("activations", output_activation)
                                                   ])


def normal_full_layer(input_layer, size, name=''):
    with tf.name_scope(name):
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size, size], name)
        b = init_bias([size], name)
        output = tf.matmul(input_layer, W) + b
        output_activation = tf.nn.relu(output, name='relu')
        return output_activation, tf.summary.merge([tf.summary.histogram("weights", W),
                                                   tf.summary.histogram("bias", b),
                                                   tf.summary.histogram("layer", output),
                                                   tf.summary.histogram("activations", output_activation)
                                                   ])


def dropout(input_layer, hold_prob, name=''):
    with tf.name_scope(name):
        return tf.nn.dropout(input_layer, keep_prob=hold_prob, name='dropout')


def reshape(input_layer, shape, name=''):
    with tf.name_scope(name):
        return tf.reshape(input_layer, shape, name='reshape')


def concat(list_tensors, name=''):
    with tf.name_scope(name):
        output = tf.concat(values=list_tensors, axis=-1, name='concat')
        return output, output.shape


def loss_fn(y_true, output_layer, name='loss_fn'):
    with tf.name_scope(name):
        loss_op = tf.losses.mean_squared_error(labels=tf.cast(y_true, tf.float32), predictions=output_layer)
        return loss_op


def metrics(y_pred, y_true, loss_op, training_type):
    y_true_bis = tf.cast(y_true, tf.int32, name='ytruebis')
    # y_pred_bis = tf.cast(y_pred, tf.int32, name='ypredbis')  #fixme: mysterious 0 gradients or 0 accuracy bug by using this line
    def loss_trn(): return tf.metrics.mean(loss_op, name='ls_train')
    def loss_cv(): return tf.metrics.mean(loss_op, name='ls_cv')
    def loss_tst(): return tf.metrics.mean(loss_op, name='ls_test')
    def acc_trn(): return tf.metrics.accuracy(labels=y_true_bis, predictions=y_pred, name='acc_train')
    def acc_cv(): return tf.metrics.accuracy(labels=y_true_bis, predictions=y_pred, name='acc_cv')
    def acc_tst(): return tf.metrics.accuracy(labels=y_true_bis, predictions=y_pred, name='acc_test')

    loss_val_op, loss_update_op = tf.case({tf.equal(training_type, 'train'): loss_trn,
                                           tf.equal(training_type, 'cv'): loss_cv},
                                           default=loss_tst)
    acc_val_op, acc_update_op = tf.case({tf.equal(training_type, 'train'): acc_trn,
                                         tf.equal(training_type, 'cv'): acc_cv},
                                         default=acc_tst)

    return tf.summary.merge([tf.summary.scalar("loss", loss_val_op)]), loss_update_op,\
           tf.summary.merge([tf.summary.scalar('accuracy', acc_val_op)]), acc_update_op

def optimizer(lr, name='AdamOptimizer'):
    with tf.name_scope(name):
        adam = tf.train.AdamOptimizer(learning_rate=lr, name='Adam')
        return adam


def train_operation(adam, gradients, name='train_op'):
    with tf.name_scope(name):
        return adam.apply_gradients(gradients, name='applyGrads')

