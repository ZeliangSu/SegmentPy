import tensorflow as tf


def init_weights(shape, name='weights'):
    return tf.get_variable('w' + name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def init_bias(shape, name='bias'):
    return tf.get_variable('b' + name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def max_pool_2by2(x, name=''):
    with tf.variable_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


def up_2by2(input_layer, name=''):
    with tf.variable_scope(name):
        return tf.image.resize_nearest_neighbor(input_layer, [2 * input_layer.shape[1], 2 * input_layer.shape[2]])


def placeholder(tensor_shape, name=''):
    with tf.variable_scope(name):
        tensor_ph = tf.placeholder(tf.float32, shape=tensor_shape, name='X_img')
        return tensor_ph, tf.shape(tensor_ph)[0]


def conv2d_layer(input_layer, shape, name='', stride=1):
    with tf.variable_scope(name):
        W = init_weights(shape, name)  # [conv_height, conv_width, in_channels, output_channels]
        b = init_bias([shape[3]], name)
        output = tf.nn.conv2d(input_layer, W, strides=[1, stride, stride, 1], padding='SAME') + b
        output_activation = tf.nn.relu(output)
        return output_activation, tf.summary.merge([tf.summary.histogram("weights", W),
                                                   tf.summary.histogram("bias", b),
                                                   tf.summary.histogram("layer", output),
                                                   tf.summary.histogram("activations", output_activation)
                                                   ])

def conv2d_transpose_layer(input_layer, shape, dyn_batch_size, stride=1, name=''):
    with tf.variable_scope(name):
        shape = [shape[0], shape[1], shape[3], shape[2]]  # switch in/output channels [height, width, output_channels, in_channels]
        W = init_weights(shape, name)
        b = init_bias([shape[2]], name)
        transpose = tf.nn.conv2d_transpose(input_layer, W, output_shape=(dyn_batch_size,
                                                                        int(input_layer.shape[1]),
                                                                        int(input_layer.shape[2]),
                                                                        int(W.shape[2])),
                                           strides=[1, stride, stride, 1], padding='SAME')
        output = transpose + b
        output_activation = tf.nn.relu(output)
        return output_activation, tf.summary.merge([tf.summary.histogram("weights", W),
                                                   tf.summary.histogram("bias", b),
                                                   tf.summary.histogram("layer", output),
                                                   tf.summary.histogram("activations", output_activation)
                                                   ])


def normal_full_layer(input_layer, size, name=''):
    with tf.variable_scope(name):
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size, size], name)
        b = init_bias([size], name)
        output = tf.matmul(input_layer, W) + b
        output_activation = tf.nn.relu(output)
        return output_activation, tf.summary.merge([tf.summary.histogram("weights", W),
                                                   tf.summary.histogram("bias", b),
                                                   tf.summary.histogram("layer", output),
                                                   tf.summary.histogram("activations", output_activation)
                                                   ])


def dropout(input_layer, hold_prob, name=''):
    with tf.variable_scope(name):
        return tf.nn.dropout(input_layer, keep_prob=hold_prob)


def reshape(input_layer, shape, name=''):
    with tf.variable_scope(name):
        return tf.reshape(input_layer, shape)


def concat(list_tensors, name=''):
    with tf.variable_scope(name):
        output = tf.concat(values=list_tensors, axis=-1)
        return output, output.shape


def loss_fn(y_true, output_layer, name=''):
    with tf.variable_scope(name):
        loss_op = tf.losses.mean_squared_error(labels=tf.cast(y_true, tf.float32), predictions=output_layer)
        return loss_op, tf.summary.merge([tf.summary.scalar("loss", loss_op)])


def cal_acc(y_pred, y_true):
    with tf.name_scope('accuracy'):
        acc = tf.reduce_mean(tf.cast(tf.equal(y_pred,
                                              tf.cast(y_true, dtype=tf.int32)), dtype=tf.float32))  #[True, False, ... True] --> [1, 0 ,...1] --> 0.667
    return acc, tf.summary.merge([tf.summary.scalar("accuracy", acc)]), tf.constant(True, dtype=tf.bool)


def optimizer(lr, name=''):
    with tf.variable_scope(name):
        adam = tf.train.AdamOptimizer(learning_rate=lr, name='AdamOptim')
        return adam


def train_operation(adam, gradients, name='train_op'):
    with tf.variable_scope(name):
        return adam.apply_gradients(gradients)

def test_operation():
    '''During test operation, one take imgs and labels for evaluating metrics from testset without minimization operation'''
    pass

def accuracy_placeholder(name='acc'):
    with tf.name_scope(name):
        acc_place = tf.placeholder(tf.float32, name='loss_sum')
        return acc_place

