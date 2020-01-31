import tensorflow as tf
import numpy as np

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)

def init_weights(shape, name='weights', reuse=False):
    """
    input:
    -------
        shape: (list) shape of the weight matrix e.g. [5, 5, 1, 32]
        name: (string) name of the node
    return:
    -------
        tensorflow variable initialized by xavier method
    """
    with tf.variable_scope(name, reuse=reuse):
        return tf.get_variable('w', shape=shape, initializer=tf.initializers.glorot_normal())


def init_bias(shape, name='bias', reuse=False):
    """
    input:
    -------
        shape: (list) shape of the bias matrix e.g. [8]
        name: (string) name of the node
    return:
    -------
        tensorflow variable initialized by xavier method
    """
    with tf.variable_scope(name, reuse=reuse):
        return tf.get_variable('b', shape=shape, initializer=tf.initializers.glorot_normal())


def max_pool_2by2(x, name=''):
    """
    input:
    -------
        x: (tf.Tensor) tensor of the previous layer
        name: (string) name of the node
    return:
    -------
        (tf.Tensor) tensor after max pooled
    """
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')


def max_pool_2by2_with_arg(x, name=''):
    with tf.name_scope(name):
        v, ind = tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
    return v, ind


def up_2by2(input_layer, name=''):
    """
    input:
    -------
        input_layer: (tf.Tensor) tensor from the previous layer
        name: (string) name of the node
    return:
    -------
        (tf.Tensor) row-by-row column-by-column copied tensor
    """
    with tf.name_scope(name):
        return tf.image.resize_nearest_neighbor(input_layer, size=[2 * input_layer.shape[1], 2 * input_layer.shape[2]], name='up')


def up_2by2_ind(input_layer, ind, name=''):
    """
    input:
    -------
        input_layer: (tf.Tensor) tensor from the previous layer
        name: (string) name of the node
    return:
    -------
        (tf.Tensor) row-by-row column-by-column copied tensor
    """
    with tf.name_scope(name):
        in_shape = input_layer.get_shape().as_list()
        # note: int64 // use tf.shape(input_layer) to get a symbolic batch size instead of Nonetype
        out_shape = [tf.cast(tf.shape(input_layer), dtype=tf.int64)[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3]]

        # prepare
        # shape: (bs, 10, 10, 640) --> (bs * 64000)
        _pool = tf.reshape(input_layer, [-1])
        _range = tf.reshape(tf.range(out_shape[0], dtype=ind.dtype), [out_shape[0], 1, 1, 1])
        # ([[1, 5, 10], [1, 5, 10]]) --> ([[1, 1, 1], [1, 1, 1]]) --> ([[0, 0, 0], [1, 1, 1]])
        tmp = tf.ones_like(ind) * _range
        # ([[0, 0, 0], [1, 1, 1]]) --> ([[0], [0], [0], [1], [1], [1]]); shp(2, 3) --> shp(6, 1)
        tmp = tf.reshape(tmp, [-1, 1])
        # shape: (bs, 10, 10, 640 / (2*2)) --> (bs * 10 * 10 * 640 / 4 , 1)
        _ind = tf.reshape(ind, [-1, 1])
        # tmp([[0], [1]]) concat axis=1 ind([[1], [6]]) --> ([[0, 1], [1, 6]]) shape: (?, 1)-->(?, 2)
        _ind = tf.concat([tmp, _ind], 1)

        # scatter
        # e.g. shp(4, 1), shp(4,), shp(1,) --> outshp(8,)
        # e.g. shp(2, 1), shp(2, 4, 4), shp(3,) --> outshp(4, 4, 4)
        # e.g. shp(300*10*10*160, 2), shp(300*10*10*160), shp(2,) --> outshp(200, 16k)
        # note: first dimension should be equal: (X, 2) and (X,), which makes the encoder and decoder symetric
        unpool = tf.scatter_nd(_ind, _pool, [out_shape[0], out_shape[1] * out_shape[2] * out_shape[3]])

        # reshape
        unpool = tf.reshape(unpool, out_shape)
        return unpool


def up_2by2_U(input_layer, dim, name=''):
    """
    input:
    -------
        input_layer: (tf.Tensor) tensor from the previous layer
        name: (string) name of the node
    return:
    -------
        (tf.Tensor) row-by-row column-by-column copied tensor
    """
    with tf.name_scope(name):
        return tf.image.resize_nearest_neighbor(input_layer, size=[2 * dim, 2 * dim], name='up')


def placeholder(tensor_shape, name=''):
    """
    input:
    -------
        tensor_shape: (list) shape of the tensorflow placeholder to hold data
        name: (string) name of the node
    return:
    -------
        tensor_ph: (tf.placeholder) tensorflow placeholder
        & the first dimension of the placeholder, alias the batch size
    """
    with tf.name_scope(name):
        tensor_ph = tf.placeholder(tf.float32, shape=tensor_shape, name='ph')
        return tensor_ph, tf.shape(tensor_ph)[0]


def conv2d_layer(input_layer, shape, stride=1, if_BN=True, is_train=None, activation='relu', name='', reuse=False):
    """
    input:
    -------
        input_layer: (tf.Tensor or tf.placeholder) tensor from the previous layer
        shape: (list) shape of the convolution layer e.g.[conv_size, conv_size, input_chan, output_chan]
        name: (string) name of the node
        stride: (int) pixel step the convolution is applied to the tensor
    return:
    -------
        output_activation: (tf.Tensor) the output the activation function of the convolution layer
        & tensorflow merged summary

    """
    with tf.name_scope(name):
        W = init_weights(shape, name, reuse=reuse)  # [conv_height, conv_width, in_channels, output_channels]
        output = tf.nn.conv2d(input_layer, W, strides=[1, stride, stride, 1], padding='SAME', name='conv')
        if 'logit' in name:
            b = init_bias([shape[3]], name, reuse=reuse)
            output_activation = tf.identity(output + b, name='identity')
            return output_activation, {name + '_W': W, name + '_b': b, name + '_activation': output_activation}
        else:
            # Batch normalization
            if if_BN:
                output = batch_norm(output, is_train=is_train, name=name + '_BN', reuse=reuse)
                output_activation = _activatioin(output, type=activation)
                return output_activation, {name + '_W': W, name + '_activation': output_activation}

            else:
                b = init_bias([shape[3]], name, reuse=reuse)
                output = output + b
                output_activation = _activatioin(output, type=activation)
                return output_activation, {name + '_W': W, name + '_b': b, name + '_activation': output_activation}


def conv2d_transpose_layer(input_layer, shape, output_shape=None, stride=1, if_BN=True, is_train=None, activation='relu', name='', reuse=False):
    """
    input:
    -------
        input_layer: (tf.Tensor or tf.placeholder) tensor from the previous layer
        shape: (list) shape of the convolution layer e.g.[conv_size, conv_size, out_chan, in_chan]
        output_shape: (int) dynamic batch size that retrieve from the encoder. In tf V12, this is not automated yet
        name: (string) name of the node
    return:
    -------
        output_activation: (tf.Tensor) the output the activation function of the convolution layer
        & tensorflow merged summary

    """
    with tf.name_scope(name):
        shape = [shape[0], shape[1], shape[3], shape[2]]  # switch in/output channels [height, width, output_channels, in_channels]
        W = init_weights(shape, name, reuse=reuse)

        # get batch_size
        dyn_input_shape = tf.shape(input_layer)
        batch_size = dyn_input_shape[0]

        # make transpose layer
        transpose = tf.nn.conv2d_transpose(input_layer, W, output_shape=[batch_size, output_shape[1], output_shape[2], output_shape[3]],
                                           strides=[1, stride, stride, 1], padding='SAME', name='transpose')

        # add activation function
        if 'logit' in name:
            b = init_bias([shape[2]], name, reuse=reuse)
            output_activation = tf.identity(transpose + b, name='identity')
            return output_activation, {name + '_W': W, name + '_b': b, name + '_activation': output_activation}
        else:
            # Batch Normalization
            if if_BN:
                output = batch_norm(transpose, is_train=is_train, name=name + '_BN', reuse=reuse)
                output_activation = _activatioin(output, type=activation)
                return output_activation, {name + '_W': W, name + '_activation': output_activation}
            else:
                b = init_bias([shape[2]], name, reuse=reuse)
                output = transpose + b
                output_activation = _activatioin(output, type=activation)
                return output_activation, {name + '_W': W, name + '_b': b, name + '_activation': output_activation}


def normal_full_layer(input_layer, size, if_BN=True, is_train=None, activation='relu', name='', reuse=False):
    """
    input:
    -------
        input_layer: (tf.Tensor or tf.placeholder) tensor from the previous layer
        size: (int) number of fully connected neuron in this layer
        name: (string) name of the node
    return:
    -------
        output_activation: (tf.Tensor) the output the activation function of the convolution layer
        & tensorflow merged summary
    """
    with tf.name_scope(name):

        # Matmul
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size, size], name, reuse=reuse)
        output = tf.matmul(input_layer, W)

        if 'logit' in name:
            b = init_bias([size], name, reuse=reuse)
            output_activation = tf.identity(output + b, name='identity')
            return output_activation, {name + '_W': W, name + '_b': b, name + '_activation': output_activation}
        else:
            # BN
            if if_BN:
                output = batch_norm(output, is_train=is_train, name=name + '_BN', reuse=reuse)
                output_activation = _activatioin(output, type=activation)
                return output_activation, {name + '_W': W, name + '_activation': output_activation}
            else:
                b = init_bias([size], name, reuse=reuse)
                output = output + b
                output_activation = _activatioin(output, type=activation)
                return output_activation, {name + '_W': W, name + '_b': b, name + '_activation': output_activation}


def dropout(input_layer, hold_prob, name=''):
    """
    input:
    -------
        input_layer: (tf.Tensor or tf.placeholder) tensor from the previous layer
        hold_prob: (float) probability of the holding
        name: (string) name of the node
    return:
    -------
        output_activation: (tf.Tensor) the output the activation function of the convolution layer
    """
    with tf.name_scope(name):
        return tf.nn.dropout(input_layer, keep_prob=hold_prob, name='dropout')


def reshape(input_layer, shape, name=''):
    """
    input:
    -------
        input_layer: (tf.Tensor or tf.placeholder) tensor from the previous layer
        shape: (list) reshape to this shape
        name: (string) name of the node
    return:
    -------
        (tf.Tensor) reshaped tensor
    """
    with tf.name_scope(name):
        return tf.reshape(input_layer, shape, name='reshape')


def flatten(input_layer, name=''):
    """
    input:
    -------
        input_layer: (tf.Tensor or tf.placeholder) tensor from the previous layer
        shape: (list) reshape to this shape
        name: (string) name of the node
    return:
    -------
        (tf.Tensor) reshaped tensor
    """
    with tf.name_scope(name):
        return tf.layers.flatten(input_layer, name='reshape')


def concat(list_tensors, name=''):
    """
    input:
    -------
        list_tensors: (list of tf.Tensor) list of tensors that to be concatenated
        name: (string) name of the node
    return:
    -------
        output: (tf.Tensor) reshaped tensor
        & shape of the output
    """
    with tf.name_scope(name):
        output = tf.concat(values=list_tensors, axis=-1, name='concat')
        return output


def MSE(y_true, logits, name='loss_fn'):
    """
    input:
    -------
        y_true: (np.ndarray? | tf.Tensor) expected results
        output_layer: (tf.Tensor) actual results
        name: (string) name of the node
    return:
    -------
        loss_op: (tf.loss?) loss operation, by default of a segmentation problem, it's appropriate to use MSE
    """
    with tf.name_scope(name):
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):  #note: use tf.get_collection(tf.GraphKeys.UPDATE_OPS) manually is better for debugging
        loss_op = tf.losses.mean_squared_error(labels=tf.cast(y_true, tf.float32), predictions=logits)
        return loss_op


def DSC(y_true, logits, name='Dice_Similarity_Coefficient'):
    # [batch_size, height, weight, class]
    axis = (1, 2, 3)
    # minimize equally for all classes (even for minor class)
    with tf.name_scope(name):
        y_true = tf.cast(y_true, tf.float32)
        numerator = 2 * tf.reduce_sum(y_true * logits, axis=axis)
        denominator = tf.reduce_sum(y_true + logits, axis=axis)
        loss_op = 1 - (numerator) / (denominator)
        return loss_op


def Cross_Entropy(y_true, logits, name='cross_entropy'):
    y_true = tf.cast(y_true, tf.float32) #(8, 512, 512, 3)
    inter = tf.log(tf.clip_by_value(logits, 1e-10, 1.0))  #(8, 512, 512, 3)
    loss = -tf.reduce_mean(y_true * inter, name=name)
    return loss


def batch_norm(input_layer, is_train, name='', reuse=False):
    '''
    :param bias_input:
    :param trainig_type: (tf.placeholder
    :param name:
    :return:
    '''
    with tf.variable_scope(name, reuse=reuse):
        return tf.layers.batch_normalization(
            input_layer,
            training=is_train,
            name='batch_norm',
        )


def metrics(y_pred, y_true, loss_op, is_training, mode='classification'):
    """
    input:
    -------
        y_pred: (tf.Tensor) output of the neural net
        y_true: (np.ndarray? | tf.Tensor) expected results
        loss_op: (tf.loss) loss function
        training_type: switching between train set, cross-validation set and test set
    return:
    -------
        merged summaries of loss and accuracy
    """
    # y_true_bis = tf.cast(y_true, tf.int32, name='ytruebis')
    if is_training:
        loss_val_op, loss_update_op = tf.metrics.mean(loss_op, name='ls_train')
        if mode == 'classification':
            y_pred = tf.cast(tf.argmax(y_pred, axis=3), tf.int32)  #[B, W, H, 1]
            y_true = tf.cast(tf.argmax(y_true, axis=3), tf.int32)  #[B, W, H, 1]
            # correct_pred = tf.equal(y_pred, y_true)
            # acc_val_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            # acc_update_op = tf.no_op(name='fake_acc_update_op')
            acc_val_op, acc_update_op = tf.metrics.accuracy(labels=y_true, predictions=y_pred, name='acc_train')
        else:
            acc_val_op, acc_update_op = tf.metrics.accuracy(labels=y_true, predictions=y_pred, name='acc_train')

    else:
        loss_val_op, loss_update_op = tf.metrics.mean(loss_op, name='ls_test')
        if mode == 'classification':
            y_pred = tf.cast(tf.argmax(y_pred, axis=3), tf.int32)
            y_true = tf.cast(tf.argmax(y_true, axis=3), tf.int32)
            acc_val_op, acc_update_op = tf.metrics.accuracy(labels=y_true, predictions=y_pred, name='acc_test')
            # correct_pred = tf.equal(y_pred, y_true)
            # acc_val_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            # acc_update_op = tf.no_op(name='fake_acc_update_op')
        else:
            acc_val_op, acc_update_op = tf.metrics.accuracy(labels=y_true, predictions=y_pred, name='acc_test')

    return tf.summary.merge([tf.summary.scalar("loss", loss_val_op)]), loss_update_op,\
           tf.summary.merge([tf.summary.scalar('accuracy', acc_val_op)]), acc_update_op


def optimizer(lr, name='AdamOptimizer'):
    """
    input:
    -------
        lr: (float) learning rate
        name: (string) name of the optimizer
    return:
    -------
        adam: (tf.AdamOptimizer?) Adam optimizer
    """
    with tf.name_scope(name):
        adam = tf.train.AdamOptimizer(learning_rate=lr, name='Adam')
        return adam


def train_operation(adam, gradients, name='train_op'):
    """
    input:
    -------
        adam: (tf.AdamOptimizer?) Adam optimizer
        gradients: (tf.Tensor?) gradients
        name: (string) name of the train operation
    return:
    -------
        (tf.Tensor) change of weights and bias from gradient descent
    """
    with tf.name_scope(name):
        return adam.apply_gradients(gradients, name='applyGrads')


def customized_softmax(inputs):
    with tf.name_scope("customized_softmax"):
        # todo: inputs = tf.clip_by_value(inputs, min=1e-10)  # to avoid exploding
        reduce_max = tf.reduce_max(inputs, axis=3, keepdims=True)  #note: ???
        # note: keepdims=True, max_axis.shape = (B, H, W, 1)
        # note: keepdims=False max_axis.shape = (B, H, W)
        nominator = tf.exp(inputs - reduce_max)  #note: here can avoid the loss becoming too big as the number of pixel increases with the number of class
                                                 # can be demonstrated easily: sum(log(small proba)_i) = inf
        # nominator = tf.exp(inputs)  #note: shape = (B, H, W, 3)
        denominator = tf.reduce_sum(nominator, axis=3, keepdims=True)  #note: shape = (B, H, W, 1)
        return nominator / denominator


def _activatioin(output, type='relu'):
    if type == 'relu':
        output_activation = tf.nn.relu(output, name='relu')
    elif type == 'sigmoid':
        output_activation = tf.nn.sigmoid(output, name='sigmoid')
    elif type == 'tanh':
        output_activation = tf.nn.tanh(output, name='tanh')
    elif type == 'leaky':
        output_activation = tf.nn.leaky_relu(output, name='leaky')
    elif '-leaky' in type:
        output_activation = tf.nn.leaky_relu(output, alpha=float(type.split('-')[0]), name='leaky')
    else:
        raise NotImplementedError('Activation function not found!')
    return output_activation


def customized_softmax_np(inputs):
    # todo: inputs = tf.clip_by_value(inputs, min=1e-10)  # to avoid exploding
    reduce_max = np.max(inputs, axis=3, keepdims=True)  #note: ???
    # note: keepdims=True, max_axis.shape = (B, H, W, 1)
    # note: keepdims=False max_axis.shape = (B, H, W)
    nominator = np.exp(inputs - reduce_max)  #note: here can avoid the loss becoming too big as the number of pixel increases with the number of class
                                             # can be demonstrated easily: sum(log(small proba)_i) = inf
    # nominator = tf.exp(inputs)  #note: shape = (B, H, W, 3)
    denominator = np.sum(nominator, axis=3, keepdims=True)  #note: shape = (B, H, W, 1)
    return nominator / denominator  #note: shape = (B, H, W, 3)
