import tensorflow as tf
from layers import *


def model_xlearn(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, activation='relu'):
    """
    xlearn segmentation convolutional neural net model with summary

    input:
    -------
        train_inputs: (tf.iterator?)
        test_inputs: (tf.iterator?)
        patch_size: (int) height and width (here we assume the same length for both)
        batch_size: (int) number of images per batch (average the gradient within a batch,
        the weights and bias upgrade after one batch)
        conv_size: (int) size of the convolution matrix e.g. 5x5, 7x7, ...
        nb_conv: (int) number of convolution per layer e.g. 32, 64, ...
        learning_rate: (float) learning rate for the optimizer
    return:
    -------
    (dictionary) dictionary of nodes in the conv net
        'y_pred': output of the neural net,
        'train_op': node of the trainning operation, once called, it will update weights and bias,
        'drop': dropout layers' probability parameters,
        'summary': merged(tensorflow) summary of histograms, evolution of scalars etc,
        'train_or_test': switch button for a training/testing input pipeline,
        'loss_update_op': node of updating loss function summary,
        'acc_update_op': node of updating accuracy summary
    """
    training_type = tf.placeholder(tf.string, name='training_type')
    drop_prob = tf.placeholder(tf.float32, name='dropout_prob')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    BN_phase = tf.placeholder_with_default(False, (), name='BN_phase')

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size
        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):

        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv], if_BN=True, is_train=BN_phase, name='conv1')#[height, width, in_channels, output_channels]
            conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], if_BN=True, is_train=BN_phase, name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], if_BN=True, is_train=BN_phase, name='conv2')
            conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=True, is_train=BN_phase, name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=True, is_train=BN_phase, name='conv3')
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=True, is_train=BN_phase, name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], activation='relu', if_BN=True, is_train=BN_phase, name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation='relu', if_BN=True, is_train=BN_phase, name='conv4bis')
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 4, 1], activation='relu', if_BN=True, is_train=BN_phase, name='conv4bisbis')

        with tf.name_scope('dnn'):
            conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
            full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 128, activation=activation, name='dnn1')
            full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
            full_layer_2, mf2 = normal_full_layer(full_dropout1, patch_size ** 2 // 128, activation=activation, name='dnn2')
            full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
            full_layer_3, mf3 = normal_full_layer(full_dropout2, patch_size ** 2 // 64, activation=activation, name='dnn3')
            full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout3')
            dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_transpose_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 4], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 4],
                                                  if_BN=True, is_train=BN_phase, name='deconv5')  #[height, width, in_channels, output_channels]
            deconv_5bis, m5b = conv2d_transpose_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 8], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 8],
                                                      if_BN=True, is_train=BN_phase, name='deconv5bis')  #fixme: strides should be 2
            concat1 = concat([up_2by2(deconv_5bis, name='up1'), conv3bis], name='concat1')  #note: up_2by2 slower than conv2d_transpose2x2

            deconv_6, m6 = conv2d_transpose_layer(concat1, [conv_size, conv_size, nb_conv * 10, nb_conv * 2], [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2],
                                                  if_BN=True, is_train=BN_phase, name='deconv6')
            deconv_6bis, m6b = conv2d_transpose_layer(deconv_6, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2],
                                                      if_BN=True, is_train=BN_phase, name='deconv6bis')  #fixme: strides should be 2
            concat2 = concat([up_2by2(deconv_6bis, name='up2'), conv2bis], name='concat2')  #note: up_2by2 slower than conv2d_transpose2x2

            deconv_7, m7 = conv2d_transpose_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2],
                                                  if_BN=True, is_train=BN_phase, name='deconv7')
            deconv_7bis, m7b = conv2d_transpose_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size //2, nb_conv * 2],
                                                      if_BN=True, is_train=BN_phase, name='deconv7bis')  #fixme: strides should be 2
            concat3 = concat([up_2by2(deconv_7bis, name='up3'), conv1bis], name='concat3')  #note: up_2by2 slower than conv2d_transpose2x2

            deconv_8, m8 = conv2d_transpose_layer(concat3, [conv_size, conv_size, nb_conv * 3, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv],
                                                  if_BN=True, is_train=BN_phase, name='deconv8')
            deconv_8bis, m8b = conv2d_transpose_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv],
                                                      if_BN=True, is_train=BN_phase, name='deconv8bis')
            logits, m8bb = conv2d_transpose_layer(deconv_8bis, [conv_size, conv_size, nb_conv, 1], [X_dyn_batsize, patch_size, patch_size, 1],
                                                  if_BN=True, is_train=BN_phase, name='logits')  #fixme: change activation function. 0 everywhere prediction?

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(lr, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)

        # train operation
        def f3():
            return opt.apply_gradients(grads, name='train_op')
        def f4():
            return tf.no_op(name='no_op')
        train_op = tf.cond(tf.equal(training_type, 'train'), lambda: f3(), lambda: f4(), name='train_cond')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        def f5():
            grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
            return tf.summary.merge([m1, m1b, m2, m2b, m3, m3b, m4, m4b, m4bb, mf1, mf2, mf3,
                                   m5, m5b, m6, m6b, m7, m7b, m8, m8b, m8bb, m_loss, m_acc, grad_sum])
        def f6():
            return tf.summary.merge([m1, m1b, m2, m2b, m3, m3b, m4, m4b, m4bb, mf1, mf2, mf3,
                                   m5, m5b, m6, m6b, m7, m7b, m8, m8b, m8bb, m_loss, m_acc])
        merged = tf.cond(tf.equal(training_type, 'train'), lambda: f5(), lambda: f6(), name='BN_cond')

    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'learning_rate': lr,
        'summary': merged,
        'BN_phase': BN_phase,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_xlearn_lite(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, activation='relu'):
    """
    lite version (less GPU occupancy) of xlearn segmentation convolutional neural net model with summary. histograms are
    saved in

    input:
    -------
        train_inputs: (tf.iterator?)
        test_inputs: (tf.iterator?)
        patch_size: (int) height and width (here we assume the same length for both)
        batch_size: (int) number of images per batch (average the gradient within a batch,
        the weights and bias upgrade after one batch)
        conv_size: (int) size of the convolution matrix e.g. 5x5, 7x7, ...
        nb_conv: (int) number of convolution per layer e.g. 32, 64, ...
        learning_rate: (float) learning rate for the optimizer
    return:
    -------
    (dictionary) dictionary of nodes in the conv net
        'y_pred': output of the neural net,
        'train_op': node of the trainning operation, once called, it will update weights and bias,
        'drop': dropout layers' probability parameters,
        'summary': compared to the original model, only summary of loss, accuracy and histograms of gradients are invovled,
        which lighten GPU resource occupancy,
        'train_or_test': switch button for a training/testing input pipeline,
        'loss_update_op': node of updating loss function summary,
        'acc_update_op': node of updating accuracy summary
    """
    training_type = tf.placeholder(tf.string, name='training_type')
    drop_prob = tf.placeholder(tf.float32, name='dropout_prob')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    BN_phase = tf.placeholder(tf.bool, name='BN_phase')

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size

        def f1(): return train_inputs

        def f2(): return test_inputs

        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv],
                                    if_BN=True, is_train=BN_phase, name='conv1')  # [height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                       if_BN=True, is_train=BN_phase, name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                    if_BN=True, is_train=BN_phase, name='conv2')
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                       if_BN=True, is_train=BN_phase, name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                    if_BN=True, is_train=BN_phase, name='conv3')
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation='relu',
                                         if_BN=True, is_train=BN_phase, name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     activation='relu', if_BN=True, is_train=BN_phase, name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         activation='relu', if_BN=True, is_train=BN_phase, name='conv4bis')
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 4, 1], activation='relu',
                                             name='conv4bisbis')

        with tf.name_scope('dnn'):
            conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
            full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 128, activation=activation,
                                                  name='dnn1', if_BN=False)
            full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
            full_layer_2, _ = normal_full_layer(full_dropout1, patch_size ** 2 // 128, activation=activation,
                                                  name='dnn2', if_BN=False)
            full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
            full_layer_3, _ = normal_full_layer(full_dropout2, patch_size ** 2 // 64, activation=activation,
                                                  name='dnn3', if_BN=False)
            full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout3')
            dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

        with tf.name_scope('decoder'):
            deconv_5, _ = conv2d_transpose_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 4],
                                                  [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 4],
                                                 if_BN=True, is_train=BN_phase, name='deconv5')  # [height, width, in_channels, output_channels]
            deconv_5bis, _ = conv2d_transpose_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                                      [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 8],
                                                    if_BN=True, is_train=BN_phase, name='deconv5bis')  # fixme: strides should be 2
            concat1 = concat([up_2by2(deconv_5bis, name='up1'), conv3bis],
                             name='concat1')  # note: up_2by2 slower than conv2d_transpose2x2

            deconv_6, _ = conv2d_transpose_layer(concat1, [conv_size, conv_size, nb_conv * 10, nb_conv * 2],
                                                  [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2],
                                                 if_BN=True, is_train=BN_phase, name='deconv6')
            deconv_6bis, _ = conv2d_transpose_layer(deconv_6, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                                      [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2],
                                                    if_BN=True, is_train=BN_phase, name='deconv6bis')  # fixme: strides should be 2
            concat2 = concat([up_2by2(deconv_6bis, name='up2'), conv2bis],
                             name='concat2')  # note: up_2by2 slower than conv2d_transpose2x2

            deconv_7, _ = conv2d_transpose_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                  [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2],
                                                 if_BN=True, is_train=BN_phase, name='deconv7')
            deconv_7bis, _ = conv2d_transpose_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                                      [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2],
                                                    if_BN=True, is_train=BN_phase, name='deconv7bis')  # fixme: strides should be 2
            concat3 = concat([up_2by2(deconv_7bis, name='up3'), conv1bis],
                             name='concat3')  # note: up_2by2 slower than conv2d_transpose2x2

            deconv_8, _ = conv2d_transpose_layer(concat3, [conv_size, conv_size, nb_conv * 3, nb_conv],
                                                  [X_dyn_batsize, patch_size, patch_size, nb_conv],
                                                 if_BN=True, is_train=BN_phase, name='deconv8')
            deconv_8bis, _ = conv2d_transpose_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv],
                                                      [X_dyn_batsize, patch_size, patch_size, nb_conv],
                                                    if_BN=True, is_train=BN_phase, name='deconv8bis')
            logits, m8bb = conv2d_transpose_layer(deconv_8bis, [conv_size, conv_size, nb_conv, 1],
                                                  [X_dyn_batsize, patch_size, patch_size, 1],
                                                  name='logits')  # fixme: change activation function. 0 everywhere prediction?

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(lr, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)

        # train operation
        def f3():
            return opt.apply_gradients(grads, name='train_op')
        def f4():
            return tf.no_op(name='no_op')
        train_op = tf.cond(tf.equal(training_type, 'train'), lambda: f3(), lambda: f4(), name='train_cond')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        def f5():
            grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
            return tf.summary.merge([m3b, m4, m4b, m4bb, mf1, m8bb, m_loss, m_acc, grad_sum])
        def f6():
            return tf.summary.merge([m3b, m4, m4b, m4bb, mf1, m8bb, m_loss, m_acc])
        merged = tf.cond(tf.equal(training_type, 'train'), lambda: f5(), lambda: f6(), name='BN_cond')

    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'learning_rate': lr,
        'summary': merged,
        'BN_phase': BN_phase,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_xlearn_custom(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, activation='relu'):
    """
    lite version (less GPU occupancy) of xlearn segmentation convolutional neural net model with summary. histograms are
    saved in

    input:
    -------
        train_inputs: (tf.iterator?)
        test_inputs: (tf.iterator?)
        patch_size: (int) height and width (here we assume the same length for both)
        batch_size: (int) number of images per batch (average the gradient within a batch,
        the weights and bias upgrade after one batch)
        conv_size: (int) size of the convolution matrix e.g. 5x5, 7x7, ...
        nb_conv: (int) number of convolution per layer e.g. 32, 64, ...
        learning_rate: (float) learning rate for the optimizer
    return:
    -------
    (dictionary) dictionary of nodes in the conv net
        'y_pred': output of the neural net,
        'train_op': node of the trainning operation, once called, it will update weights and bias,
        'drop': dropout layers' probability parameters,
        'summary': compared to the original model, only summary of loss, accuracy and histograms of gradients are invovled,
        which lighten GPU resource occupancy,
        'train_or_test': switch button for a training/testing input pipeline,
        'loss_update_op': node of updating loss function summary,
        'acc_update_op': node of updating accuracy summary
    """
    training_type = tf.placeholder(tf.string, name='training_type')
    drop_prob = tf.placeholder(tf.float32, name='dropout_prob')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size
        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv],
                                     name='conv1')  # [height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], name='conv2')
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                     name='conv3')
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation='relu', name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     activation='leaky', name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         activation='leaky', name='conv4bis')
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 4, 1], activation='0.5-leaky',
                                             name='conv4bisbis')

        with tf.name_scope('dnn'):
            conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
            full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 128, activation=activation,
                                                  name='dnn1')
            full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
            full_layer_2, _ = normal_full_layer(full_dropout1, patch_size ** 2 // 128, activation=activation,
                                                  name='dnn2')
            full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
            full_layer_3, _ = normal_full_layer(full_dropout2, patch_size ** 2 // 64, activation=activation,
                                                  name='dnn3')
            full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout3')
            dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

        with tf.name_scope('decoder'):
            concat0 = concat([dnn_reshape, conv4bisbis], name='concat0')
            deconv_5, _ = conv2d_transpose_layer(concat0, [conv_size, conv_size, 2, nb_conv * 4],
                                                  [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 4],
                                                  name='deconv5')  # [height, width, in_channels, output_channels]
            deconv_5bis, _ = conv2d_transpose_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                                      [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 8],
                                                      name='deconv5bis')  # fixme: strides should be 2
            concat1 = concat([up_2by2(deconv_5bis, name='up1'), conv3bis],
                             name='concat1')  # note: up_2by2 slower than conv2d_transpose2x2

            deconv_6, _ = conv2d_transpose_layer(concat1, [conv_size, conv_size, nb_conv * 10, nb_conv * 2],
                                                  [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2],
                                                  name='deconv6')
            deconv_6bis, _ = conv2d_transpose_layer(deconv_6, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                                      [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2],
                                                      name='deconv6bis')  # fixme: strides should be 2
            concat2 = concat([up_2by2(deconv_6bis, name='up2'), conv2bis],
                             name='concat2')  # note: up_2by2 slower than conv2d_transpose2x2

            deconv_7, _ = conv2d_transpose_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                  [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2],
                                                  name='deconv7')
            deconv_7bis, _ = conv2d_transpose_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                                      [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2],
                                                      name='deconv7bis')  # fixme: strides should be 2
            concat3 = concat([up_2by2(deconv_7bis, name='up3'), conv1bis],
                             name='concat3')  # note: up_2by2 slower than conv2d_transpose2x2

            deconv_8, _ = conv2d_transpose_layer(concat3, [conv_size, conv_size, nb_conv * 3, nb_conv],
                                                  [X_dyn_batsize, patch_size, patch_size, nb_conv], name='deconv8')
            deconv_8bis, _ = conv2d_transpose_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv],
                                                      [X_dyn_batsize, patch_size, patch_size, nb_conv],
                                                      name='deconv8bis')
            logits, m8bb = conv2d_transpose_layer(deconv_8bis, [conv_size, conv_size, nb_conv, 1],
                                                  [X_dyn_batsize, patch_size, patch_size, 1],
                                                  name='logits')  # fixme: change activation function. 0 everywhere prediction?

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(lr, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)

        # train operation
        def f3():
            return opt.apply_gradients(grads, name='train_op')
        def f4():
            return tf.no_op(name='no_op')
        train_op = tf.cond(tf.equal(training_type, 'train'), lambda: f3(), lambda: f4(), name='train_cond')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        def f5():
            grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
            return tf.summary.merge([m3b, m4, m4b, m4bb, mf1, m8bb, m_loss, m_acc, grad_sum])
        def f6():
            return tf.summary.merge([m3b, m4, m4b, m4bb, mf1, m8bb, m_loss, m_acc])
        merged = tf.cond(tf.equal(training_type, 'train'), lambda: f5(), lambda: f6(), name='BN_cond')

    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'learning_rate': lr,
        'summary': merged,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_LRCS_custom(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, activation='relu'):
    """
    lite version (less GPU occupancy) of xlearn segmentation convolutional neural net model with summary. histograms are
    saved in

    input:
    -------
        train_inputs: (tf.iterator?)
        test_inputs: (tf.iterator?)
        patch_size: (int) height and width (here we assume the same length for both)
        batch_size: (int) number of images per batch (average the gradient within a batch,
        the weights and bias upgrade after one batch)
        conv_size: (int) size of the convolution matrix e.g. 5x5, 7x7, ...
        nb_conv: (int) number of convolution per layer e.g. 32, 64, ...
        learning_rate: (float) learning rate for the optimizer
    return:
    -------
    (dictionary) dictionary of nodes in the conv net
        'y_pred': output of the neural net,
        'train_op': node of the trainning operation, once called, it will update weights and bias,
        'drop': dropout layers' probability parameters,
        'summary': compared to the original model, only summary of loss, accuracy and histograms of gradients are invovled,
        which lighten GPU resource occupancy,
        'train_or_test': switch button for a training/testing input pipeline,
        'loss_update_op': node of updating loss function summary,
        'acc_update_op': node of updating accuracy summary
    """
    training_type = tf.placeholder(tf.string, name='training_type')
    # drop_prob = tf.placeholder(tf.float32, name='dropout_prob')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    # X_dyn_batsize = batch_size

    with tf.name_scope('input_pipeline'):

        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv], activation=activation,
                                    name='conv1')  # [height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], activation=activation, name='conv1bis')
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], activation=activation, name='conv2')
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation=activation, name='conv2bis')
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], activation=activation, name='conv3')
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation=activation, name='conv3bis')
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     activation=activation, name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                         activation=activation, name='conv4bis')
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 8, 1], activation=activation,
                                             name='conv4bisbis')

        with tf.name_scope('dnn'):
            conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
            full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 128, activation='relu',
                                                  name='dnn1')
            full_layer_2, _ = normal_full_layer(full_layer_1, patch_size ** 2 // 128, activation='relu',
                                                name='dnn2')
            full_layer_3, _ = normal_full_layer(full_layer_2, patch_size ** 2 // 64, activation='relu',
                                                name='dnn3')
            dnn_reshape = reshape(full_layer_3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

        with tf.name_scope('decoder'):
            deconv_5, _ = conv2d_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 8], activation=activation,
                                       name='deconv5')  # [height, width, in_channels, output_channels]
            deconv_5bis, _ = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                          activation=activation, name='deconv5bis')

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                       activation=activation, name='deconv6')
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                          activation=activation, name='deconv6bis')

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                       activation=activation, name='deconv7')
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                          activation=activation, name='deconv7bis')

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv, nb_conv],activation=activation, name='deconv8')
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv],
                                          activation=activation, name='deconv8bis')
            logits, m8bb = conv2d_layer(deconv_8bis, [conv_size, conv_size, nb_conv, 1], name='logits')

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(lr, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)

        # train operation
        def f3():
            return opt.apply_gradients(grads, name='train_op')
        def f4():
            return tf.no_op(name='no_op')
        train_op = tf.cond(tf.equal(training_type, 'train'), lambda: f3(), lambda: f4(), name='train_cond')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        def f5():
            grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
            return tf.summary.merge([m3b, m4, m4b, m4bb, mf1, m8bb, m_loss, m_acc, grad_sum])
        def f6():
            return tf.summary.merge([m3b, m4, m4b, m4bb, mf1, m8bb, m_loss, m_acc])
        merged = tf.cond(tf.equal(training_type, 'train'), lambda: f5(), lambda: f6(), name='BN_cond')

    return {
        'y_pred': logits,
        'train_op': train_op,
        'learning_rate': lr,
        'summary': merged,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_Unet(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, activation='relu'):
    """
    xlearn segmentation convolutional neural net model with summary

    input:
    -------
        train_inputs: (tf.iterator?)
        test_inputs: (tf.iterator?)
        patch_size: (int) height and width (here we assume the same length for both)
        batch_size: (int) number of images per batch (average the gradient within a batch,
        the weights and bias upgrade after one batch)
        conv_size: (int) size of the convolution matrix e.g. 5x5, 7x7, ...
        nb_conv: (int) number of convolution per layer e.g. 32, 64, ...
        learning_rate: (float) learning rate for the optimizer
    return:
    -------
    (dictionary) dictionary of nodes in the conv net
        'y_pred': output of the neural net,
        'train_op': node of the trainning operation, once called, it will update weights and bias,
        'drop': dropout layers' probability parameters,
        'summary': merged(tensorflow) summary of histograms, evolution of scalars etc,
        'train_or_test': switch button for a training/testing input pipeline,
        'loss_update_op': node of updating loss function summary,
        'acc_update_op': node of updating accuracy summary
    """
    training_type = tf.placeholder(tf.string, name='training_type')
    drop_prob = tf.placeholder(tf.float32, name='dropout_prob')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size
        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):

        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv], activation=activation, name='conv1')#[height, width, in_channels, output_channels]
            conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], activation=activation, name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], activation=activation, name='conv2')
            conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation=activation, name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], activation=activation, name='conv3')
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation=activation, name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8], activation=activation, name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8], activation=activation, name='conv4bis')
            conv4_pooling = max_pool_2by2(conv4bis, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16], activation=activation, name='bot5')
            conv5bis, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16], activation=activation, name='bot5bis')
            deconv1, m5u = conv2d_transpose_layer(conv5bis, [conv_size, conv_size, nb_conv * 16, nb_conv * 8], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 8], stride=2, activation=activation, name='deconv1')

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4bis], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8], activation=activation, name='conv6')  #[height, width, in_channels, output_channels]
            conv_6bis, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 8], activation=activation, name='conv6bis')
            deconv2, m6u = conv2d_transpose_layer(conv_6bis, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], [X_dyn_batsize, patch_size // 4, patch_size //4, nb_conv * 4], stride=2, activation=activation, dropout=drop_prob, name='deconv2')

            concat2 = concat([deconv2, conv3bis], name='concat2')
            conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], activation=activation, name='conv7')
            conv_7bis, m7b = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation=activation, name='conv7bis')
            deconv3, m7u = conv2d_transpose_layer(conv_7bis, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2], stride=2, activation=activation, dropout=drop_prob, name='deconv3')

            concat3 = concat([deconv3, conv2bis], name='concat3')
            conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], activation=activation, name='conv8')
            conv_8bis, m8b = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation=activation,  name='conv8bis')
            deconv4, m8u = conv2d_transpose_layer(conv_8bis, [conv_size, conv_size, nb_conv * 2, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv], stride=2, activation=activation, dropout=drop_prob, name='deconv4')

            concat4 = concat([deconv4, conv1bis], name='concat4')
            deconv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv], activation=activation, name='conv9')
            deconv_9bis, m9b = conv2d_layer(deconv_9, [conv_size, conv_size, nb_conv, nb_conv], activation=activation, name='conv9bis')
            logits, m9b = conv2d_layer(deconv_9bis, [conv_size, conv_size, nb_conv, 1], activation=activation, name='logits')

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(lr, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)

        # train operation
        def f3():
            return opt.apply_gradients(grads, name='train_op')
        def f4():
            return tf.no_op(name='no_op')
        train_op = tf.cond(tf.equal(training_type, 'train'), lambda: f3(), lambda: f4(), name='train_cond')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        def f5():
            grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
            return tf.summary.merge([m1, m1b, m2, m2b, m3, m3b, m4, m4b, m5, m5b, m5u,
                                   m6, m6b, m6u, m7, m7b, m7u, m8, m8b, m8u, m9, m9b, m_loss, m_acc,
                                     grad_sum])
        def f6():
            return tf.summary.merge([m1, m1b, m2, m2b, m3, m3b, m4, m4b, m5, m5b, m5u,
                                   m6, m6b, m6u, m7, m7b, m7u, m8, m8b, m8u, m9, m9b, m_loss, m_acc,
                                     m_acc])
        merged = tf.cond(tf.equal(training_type, 'train'), lambda: f5(), lambda: f6(), name='BN_cond')

    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'learning_rate': lr,
        'summary': merged,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_Unet_lite(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, activation='relu'):
    """
    xlearn segmentation convolutional neural net model with summary

    input:
    -------
        train_inputs: (tf.iterator?)
        test_inputs: (tf.iterator?)
        patch_size: (int) height and width (here we assume the same length for both)
        batch_size: (int) number of images per batch (average the gradient within a batch,
        the weights and bias upgrade after one batch)
        conv_size: (int) size of the convolution matrix e.g. 5x5, 7x7, ...
        nb_conv: (int) number of convolution per layer e.g. 32, 64, ...
        learning_rate: (float) learning rate for the optimizer
    return:
    -------
    (dictionary) dictionary of nodes in the conv net
        'y_pred': output of the neural net,
        'train_op': node of the trainning operation, once called, it will update weights and bias,
        'drop': dropout layers' probability parameters,
        'summary': merged(tensorflow) summary of histograms, evolution of scalars etc,
        'train_or_test': switch button for a training/testing input pipeline,
        'loss_update_op': node of updating loss function summary,
        'acc_update_op': node of updating accuracy summary
    """
    training_type = tf.placeholder(tf.string, name='training_type')
    drop_prob = tf.placeholder(tf.float32, name='dropout_prob')
    lr = tf.placeholder(tf.float32, name='learning_rate')

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size
        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):

        with tf.name_scope('contractor'):
            conv1, _ = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv], activation=activation, name='conv1')#[height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], activation=activation, name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], activation=activation, name='conv2')
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation=activation, name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], activation=activation, name='conv3')
            conv3bis, _ = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation=activation, name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8], activation=activation, name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8], activation=activation, name='conv4bis')
            conv4_pooling = max_pool_2by2(conv4bis, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16], activation=activation, name='bot5')
            conv5bis, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16], activation=activation, name='bot5bis')
            deconv1, m5u = conv2d_transpose_layer(conv5bis, [conv_size, conv_size, nb_conv * 16, nb_conv * 8], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 8], stride=2, activation=activation, name='deconv1')

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4bis], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8], activation=activation, name='conv6')  #[height, width, in_channels, output_channels]
            conv_6bis, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 8], activation=activation, name='conv6bis')
            deconv2, m6u = conv2d_transpose_layer(conv_6bis, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 4], stride=2, activation=activation, name='deconv2')

            concat2 = concat([deconv2, conv3bis], name='concat2')
            conv_7, _ = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], activation=activation, name='conv7')
            conv_7bis, _ = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation=activation, name='conv7bis')
            deconv3, _ = conv2d_transpose_layer(conv_7bis, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2], stride=2, activation=activation, name='deconv3')

            concat3 = concat([deconv3, conv2bis], name='concat3')
            conv_8, _ = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], activation=activation, name='conv8')
            conv_8bis, _ = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation=activation,  name='conv8bis')
            deconv4, _ = conv2d_transpose_layer(conv_8bis, [conv_size, conv_size, nb_conv * 2, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv], stride=2, activation=activation, name='deconv4')

            concat4 = concat([deconv4, conv1bis], name='concat4')
            deconv_9, _ = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv], activation=activation, name='conv9')
            deconv_9bis, _ = conv2d_layer(deconv_9, [conv_size, conv_size, nb_conv, nb_conv], activation=activation, name='conv9bis')
            logits, m9b = conv2d_layer(deconv_9bis, [conv_size, conv_size, nb_conv, 1], activation=activation, name='logits')

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(lr, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)

        # train operation
        def f3():
            return opt.apply_gradients(grads, name='train_op')
        def f4():
            return tf.no_op(name='no_op')
        train_op = tf.cond(tf.equal(training_type, 'train'), lambda: f3(), lambda: f4(), name='train_cond')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        def f5():
            grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
            return tf.summary.merge([m4, m4b, m5, m5b, m5u,
                                   m6, m6b, m6u, m9b, m_loss, m_acc, grad_sum])
        def f6():
            return tf.summary.merge([m4, m4b, m5, m5b, m5u,
                                   m6, m6b, m6u, m9b, m_loss, m_acc])
        merged = tf.cond(tf.equal(training_type, 'train'), lambda: f5(), lambda: f6(), name='BN_cond')

    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'learning_rate': lr,
        'summary': merged,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op,
    }

