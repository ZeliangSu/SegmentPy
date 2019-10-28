import tensorflow as tf


def init_weights(shape, name='weights'):
    """
    input:
    -------
        shape: (list) shape of the weight matrix e.g. [5, 5, 1, 32]
        name: (string) name of the node
    return:
    -------
        tensorflow variable initialized by xavier method
    """
    with tf.variable_scope(name):
        return tf.get_variable('w', shape=shape, initializer=tf.contrib.layers.xavier_initializer())


def init_bias(shape, name='bias'):
    """
    input:
    -------
        shape: (list) shape of the bias matrix e.g. [8]
        name: (string) name of the node
    return:
    -------
        tensorflow variable initialized by xavier method
    """
    with tf.variable_scope(name):
        return tf.get_variable('b', shape=shape, initializer=tf.contrib.layers.xavier_initializer())


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


def conv2d_layer(input_layer, shape, stride=1, activation='relu', dropout=1, name=''):
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
        W = init_weights(shape, name)  # [conv_height, conv_width, in_channels, output_channels]
        b = init_bias([shape[3]], name)
        output = tf.nn.conv2d(input_layer, W, strides=[1, stride, stride, 1], padding='SAME', name='conv') + b
        if dropout != 1:
            output = tf.nn.dropout(output, keep_prob=dropout)

        if name == 'logits':
            output_activation = output
        else:
            if activation == 'relu':
                output_activation = tf.nn.relu(output, name='relu')
            elif activation == 'sigmoid':
                output_activation = tf.nn.sigmoid(output, name='sigmoid')
            elif activation == 'tanh':
                output_activation = tf.nn.tanh(output, name='tanh')
            elif activation == 'leaky':
                output_activation = tf.nn.leaky_relu(output, name='leaky')
            elif '-leaky' in activation:
                output_activation = tf.nn.leaky_relu(output, alpha=float(activation.split('-')[0]), name='leaky')
            else:
                raise NotImplementedError('Activation function not found!')
        return output_activation, tf.summary.merge([tf.summary.histogram("weights", W),
                                                   tf.summary.histogram("bias", b),
                                                   tf.summary.histogram("layer", output),
                                                   tf.summary.histogram("activations", output_activation)
                                                   ])


def conv2d_transpose_layer(input_layer, shape, output_shape=None, stride=1, activation='relu', dropout=1, name=''):
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
        W = init_weights(shape, name)
        b = init_bias([shape[2]], name)
        dyn_input_shape = tf.shape(input_layer)
        batch_size = dyn_input_shape[0]
        output_shape = tf.stack([batch_size, output_shape[1], output_shape[2], output_shape[3]])
        transpose = tf.nn.conv2d_transpose(input_layer, W, output_shape=output_shape,
                                           strides=[1, stride, stride, 1], padding='SAME', name=name)
        if dropout != 1:
            transpose = tf.nn.dropout(transpose, keep_prob=dropout)
        output = transpose + b

        if name == 'logits':
            output_activation = output
        else:
            if activation == 'relu':
                output_activation = tf.nn.relu(output, name='relu')
            elif activation == 'sigmoid':
                output_activation = tf.nn.sigmoid(output, name='sigmoid')
            elif activation == 'tanh':
                output_activation = tf.nn.tanh(output, name='tanh')
            elif activation == 'leaky':
                output_activation = tf.nn.leaky_relu(output, name='leaky')
            elif '-leaky' in activation:
                output_activation = tf.nn.leaky_relu(output, alpha=float(activation.split('-')[0]), name='leaky')
            else:
                raise NotImplementedError('Activation function not found!')
        return output_activation, tf.summary.merge([tf.summary.histogram("weights", W),
                                                   tf.summary.histogram("bias", b),
                                                   tf.summary.histogram("layer", output),
                                                   tf.summary.histogram("activations", output_activation)
                                                   ])


def normal_full_layer(input_layer, size, activation='relu', name=''):
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
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size, size], name)
        b = init_bias([size], name)
        output = tf.matmul(input_layer, W) + b
        if activation == 'relu':
            output_activation = tf.nn.relu(output, name='relu')
        elif activation == 'sigmoid':
            output_activation = tf.nn.sigmoid(output, name='sigmoid')
        elif activation == 'tanh':
            output_activation = tf.nn.tanh(output, name='tanh')
        elif activation == 'leaky':
            output_activation = tf.nn.leaky_relu(output, name='leaky')
        else:
            raise NotImplementedError('Activation function not found!')
        return output_activation, tf.summary.merge([tf.summary.histogram("weights", W),
                                                   tf.summary.histogram("bias", b),
                                                   tf.summary.histogram("layer", output),
                                                   tf.summary.histogram("activations", output_activation)
                                                   ])


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


def loss_fn(y_true, output_layer, name='loss_fn'):
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
        loss_op = tf.losses.mean_squared_error(labels=tf.cast(y_true, tf.float32), predictions=output_layer)
        return loss_op


def metrics(y_pred, y_true, loss_op, training_type):
    """
    input:
    -------
        y_pred: (tf.Tensor) output of the neural net
        y_true: (np.ndarray? | tf.Tensor) expected results
        loss_op: (tf.loss) loss function
        training_type: switching between train set, cross-validation set and test set#fixme: should I change the name?
    return:
    -------
        merged summaries of loss and accuracy
    """
    y_true_bis = tf.cast(y_true, tf.int32, name='ytruebis')  #fixme: mysterious 0 gradients or 0 accuracy bug by using this line
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

