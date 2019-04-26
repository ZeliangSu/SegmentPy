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
        output = tf.nn.conv2d(input_layer, W, strides=[1, stride, stride, 1], padding='SAME', name='deconv') + b
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
                                           strides=[1, stride, stride, 1], padding='SAME', name='conv')
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
        return loss_op, tf.summary.merge([tf.summary.scalar("loss", tf.metrics.mean(loss_op))])


def cal_acc(y_pred, y_true, name='accuracy'):
    with tf.name_scope('accuracy'):
        # acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(y_pred, dtype=tf.int32),
        #                                       tf.cast(y_true, dtype=tf.int32)), dtype=tf.float32), name=name)  #[True, False, ... True] --> [1, 0 ,...1] --> 0.667
        # return acc, tf.summary.merge([tf.summary.scalar("accuracy", acc)])
        return tf.summary.merge([tf.summary.scalar('accuracy', tf.metrics.accuracy(labels=y_true, predictions=y_pred))])

def optimizer(lr, name='AdamOptimizer'):
    with tf.name_scope(name):
        adam = tf.train.AdamOptimizer(learning_rate=lr, name='Adam')
        return adam


def train_operation(adam, gradients, name='train_op'):
    with tf.name_scope(name):
        return adam.apply_gradients(gradients, name='applyGrads')

