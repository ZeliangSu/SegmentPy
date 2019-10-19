import tensorflow as tf
from layers import *


def model_xlearn(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, learning_rate=0.0001, activation='relu'):
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

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size
        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')
        # inputs = tf.identity(inputs, name='identity')  #note: fix a bug of dimension in tflite inference

    with tf.name_scope('model'):

        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv], name='conv1')#[height, width, in_channels, output_channels]
            conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], name='conv2')
            conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv3')
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], activation='relu', name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation='relu', name='conv4bis')
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 4, 1], activation='relu', name='conv4bisbis')

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
            deconv_5, m5 = conv2d_transpose_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 4], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 4], name='deconv5')  #[height, width, in_channels, output_channels]
            deconv_5bis, m5b = conv2d_transpose_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 8], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 8],name='deconv5bis')  #fixme: strides should be 2
            concat1 = concat([up_2by2(deconv_5bis, name='up1'), conv3bis], name='concat1')  #note: up_2by2 slowing than conv2d_transpose2x2

            deconv_6, m6 = conv2d_transpose_layer(concat1, [conv_size, conv_size, nb_conv * 10, nb_conv * 2], [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2], name='deconv6')
            deconv_6bis, m6b = conv2d_transpose_layer(deconv_6, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2], name='deconv6bis')  #fixme: strides should be 2
            concat2 = concat([up_2by2(deconv_6bis, name='up2'), conv2bis], name='concat2')  #note: up_2by2 slowing than conv2d_transpose2x2

            deconv_7, m7 = conv2d_transpose_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2], name='deconv7')
            deconv_7bis, m7b = conv2d_transpose_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size //2, nb_conv * 2], name='deconv7bis')  #fixme: strides should be 2
            concat3 = concat([up_2by2(deconv_7bis, name='up3'), conv1bis], name='concat3')  #note: up_2by2 slowing than conv2d_transpose2x2

            deconv_8, m8 = conv2d_transpose_layer(concat3, [conv_size, conv_size, nb_conv * 3, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv], name='deconv8')
            deconv_8bis, m8b = conv2d_transpose_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv], name='deconv8bis')
            logits, m8bb = conv2d_transpose_layer(deconv_8bis, [conv_size, conv_size, nb_conv, 1], [X_dyn_batsize, patch_size, patch_size, 1], name='logits')  #fixme: change activation function. 0 everywhere prediction?

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(learning_rate, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)
        grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])

        # train operation
        train_op = opt.apply_gradients(grads, name='train_op')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        merged = tf.summary.merge([m1, m1b, m2, m2b, m3, m3b, m4, m4b, m4bb, mf1, mf2, mf3,
                                   m5, m5b, m6, m6b, m7, m7b, m8, m8b, m8bb, m_loss, m_acc, grad_sum])  #fixme: withdraw summary of histories for GPU resource reason
    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'summary': merged,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_xlearn_lite(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, learning_rate=0.0001):
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

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size
        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):

        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv], name='conv1')#[height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], name='conv2')
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv3')
            conv3bis, _ = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, _ = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], name='conv4')
            conv4bis, _ = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], name='conv4bis')
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 4, 1], name='conv4bisbis')

        with tf.name_scope('dnn'):
            conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
            full_layer_1, _ = normal_full_layer(conv4_flat, patch_size ** 2 // 128, name='dnn1')
            full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
            full_layer_2, _ = normal_full_layer(full_dropout1, patch_size ** 2 // 128, name='dnn2')
            full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
            full_layer_3, _ = normal_full_layer(full_dropout2, patch_size ** 2 // 64, name='dnn3')
            full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout3')
            dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

        with tf.name_scope('decoder'):
            deconv_5, _ = conv2d_transpose_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 4], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 4], name='deconv5')  #[height, width, in_channels, output_channels]
            deconv_5bis, _ = conv2d_transpose_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 8], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 8], stride=1, name='deconv5bis')  #fixme: strides should be 2
            concat1 = concat([up_2by2(deconv_5bis, name='up1'), conv3bis], name='concat1')

            deconv_6, _ = conv2d_transpose_layer(concat1, [conv_size, conv_size, nb_conv * 10, nb_conv * 2], [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2], name='deconv6')
            deconv_6bis, _ = conv2d_transpose_layer(deconv_6, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 2], stride=1, name='deconv6bis')  #fixme: strides should be 2
            concat2 = concat([up_2by2(deconv_6bis, name='up2'), conv2bis], name='concat2')

            deconv_7, _ = conv2d_transpose_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2], name='deconv7')
            deconv_7bis, _ = conv2d_transpose_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2], stride=1, name='deconv7bis')  #fixme: strides should be 2
            concat3 = concat([up_2by2(deconv_7bis, name='up3'), conv1bis], name='concat3')

            deconv_8, _ = conv2d_transpose_layer(concat3, [conv_size, conv_size, nb_conv * 3, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv], name='deconv8')
            deconv_8bis, _ = conv2d_transpose_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv], name='deconv8bis')
            logits, _ = conv2d_transpose_layer(deconv_8bis, [conv_size, conv_size, nb_conv, 1], [X_dyn_batsize, patch_size, patch_size, 1], name='deconv8bisbis')  #fixme: change activation function. 0 everywhere prediction?

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(learning_rate, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)
        grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])

        # train operation
        train_op = opt.apply_gradients(grads, name='train_op')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        merged = tf.summary.merge([m_loss, m_acc, grad_sum])

    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'summary': merged,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_Unet(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, learning_rate=0.0001, activation='relu'):
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

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size
        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):

        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv], activation=activation, dropout=drop_prob, name='conv1')#[height, width, in_channels, output_channels]
            conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], activation=activation, dropout=drop_prob, name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], activation=activation, dropout=drop_prob, name='conv2')
            conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation=activation, dropout=drop_prob, name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], activation=activation, dropout=drop_prob, name='conv3')
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation=activation, dropout=drop_prob, name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8], activation=activation, dropout=drop_prob, name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8], activation=activation, dropout=drop_prob, name='conv4bis')
            conv4_pooling = max_pool_2by2(conv4bis, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16], activation=activation, name='bot5')
            conv5bis, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16], activation=activation, name='bot5bis')
            deconv1, m5u = conv2d_transpose_layer(conv5bis, [conv_size, conv_size, nb_conv * 16, nb_conv * 8], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 8], stride=2, activation=activation, name='deconv1')

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4bis], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8], activation=activation, dropout=drop_prob, name='conv6')  #[height, width, in_channels, output_channels]
            conv_6bis, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 8], activation=activation, dropout=drop_prob, name='conv6bis')
            deconv2, m6u = conv2d_transpose_layer(conv_6bis, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], [X_dyn_batsize, patch_size // 4, patch_size //4, nb_conv * 4], stride=2, activation=activation, dropout=drop_prob, name='deconv2')

            concat2 = concat([deconv2, conv3bis], name='concat2')
            conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], activation=activation, dropout=drop_prob, name='conv7')
            conv_7bis, m7b = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation=activation, dropout=drop_prob, name='conv7bis')
            deconv3, m7u = conv2d_transpose_layer(conv_7bis, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2], stride=2, activation=activation, dropout=drop_prob, name='deconv3')

            concat3 = concat([deconv3, conv2bis], name='concat3')
            conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], activation=activation, dropout=drop_prob, name='conv8')
            conv_8bis, m8b = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation=activation, dropout=drop_prob, name='conv8bis')
            deconv4, m8u = conv2d_transpose_layer(conv_8bis, [conv_size, conv_size, nb_conv * 2, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv], stride=2, activation=activation, dropout=drop_prob, name='deconv4')

            concat4 = concat([deconv4, conv1bis], name='concat4')
            deconv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv], activation=activation, dropout=drop_prob, name='conv9')
            deconv_9bis, m9b = conv2d_layer(deconv_9, [conv_size, conv_size, nb_conv, nb_conv], activation=activation, dropout=drop_prob, name='conv9bis')
            logits, m9b = conv2d_layer(deconv_9bis, [conv_size, conv_size, nb_conv, 1], activation=activation, name='logits')

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(learning_rate, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)
        grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])

        # train operation
        train_op = opt.apply_gradients(grads, name='train_op')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        merged = tf.summary.merge([m1, m1b, m2, m2b, m3, m3b, m4, m4b, m5, m5b, m5u,
                                   m6, m6b, m6u, m7, m7b, m7u, m8, m8b, m8u, m9, m9b, m_loss, m_acc, grad_sum])  #fixme: withdraw summary of histories for GPU resource reason
    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'summary': merged,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_Unet_lite(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, learning_rate=0.0001, activation='relu'):
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

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size
        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):

        with tf.name_scope('contractor'):
            conv1, _ = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv], activation=activation, dropout=drop_prob, name='conv1')#[height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], activation=activation, dropout=drop_prob, name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], activation=activation, dropout=drop_prob, name='conv2')
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation=activation, dropout=drop_prob, name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], activation=activation, dropout=drop_prob, name='conv3')
            conv3bis, _ = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation=activation, dropout=drop_prob, name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8], activation=activation, dropout=drop_prob, name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8], activation=activation, dropout=drop_prob, name='conv4bis')
            conv4_pooling = max_pool_2by2(conv4bis, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16], activation=activation, name='bot5')
            conv5bis, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16], activation=activation, name='bot5bis')
            deconv1, m5u = conv2d_transpose_layer(conv5bis, [conv_size, conv_size, nb_conv * 16, nb_conv * 8], [X_dyn_batsize, patch_size // 8, patch_size // 8, nb_conv * 8], stride=2, activation=activation, name='deconv1')

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4bis], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8], activation=activation, dropout=drop_prob, name='conv6')  #[height, width, in_channels, output_channels]
            conv_6bis, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 8], activation=activation, dropout=drop_prob, name='conv6bis')
            deconv2, m6u = conv2d_transpose_layer(conv_6bis, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], [X_dyn_batsize, patch_size // 4, patch_size // 4, nb_conv * 4], stride=2, activation=activation, dropout=drop_prob, name='deconv2')

            concat2 = concat([deconv2, conv3bis], name='concat2')
            conv_7, _ = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], activation=activation, dropout=drop_prob, name='conv7')
            conv_7bis, _ = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], activation=activation, dropout=drop_prob, name='conv7bis')
            deconv3, _ = conv2d_transpose_layer(conv_7bis, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], [X_dyn_batsize, patch_size // 2, patch_size // 2, nb_conv * 2], stride=2, activation=activation, dropout=drop_prob, name='deconv3')

            concat3 = concat([deconv3, conv2bis], name='concat3')
            conv_8, _ = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], activation=activation, dropout=drop_prob, name='conv8')
            conv_8bis, _ = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], activation=activation, dropout=drop_prob, name='conv8bis')
            deconv4, _ = conv2d_transpose_layer(conv_8bis, [conv_size, conv_size, nb_conv * 2, nb_conv], [X_dyn_batsize, patch_size, patch_size, nb_conv], stride=2, activation=activation, dropout=drop_prob, name='deconv4')

            concat4 = concat([deconv4, conv1bis], name='concat4')
            deconv_9, _ = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv], activation=activation, dropout=drop_prob, name='conv9')
            deconv_9bis, _ = conv2d_layer(deconv_9, [conv_size, conv_size, nb_conv, nb_conv], activation=activation, dropout=drop_prob, name='conv9bis')
            logits, m9b = conv2d_layer(deconv_9bis, [conv_size, conv_size, nb_conv, 1], activation=activation, name='logits')

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(learning_rate, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)
        grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])

        # train operation
        train_op = opt.apply_gradients(grads, name='train_op')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        merged = tf.summary.merge([m4, m4b, m5, m5b, m5u,
                                   m6, m6b, m6u, m9b, m_loss, m_acc, grad_sum])  #fixme: withdraw summary of histories for GPU resource reason
    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'summary': merged,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_improved(train_inputs, test_inputs, patch_size, batch_size, conv_size, nb_conv, learning_rate=0.0001):
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

    with tf.name_scope('input_pipeline'):
        X_dyn_batsize = batch_size
        def f1(): return train_inputs
        def f2(): return test_inputs
        inputs = tf.cond(tf.equal(training_type, 'test'), lambda: f2(), lambda: f1(), name='input_cond')

    with tf.name_scope('model'):

        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(inputs['img'], shape=[conv_size, conv_size, 1, nb_conv], name='conv1')#[height, width, in_channels, output_channels]
            conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], name='conv1bis')
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], name='conv2')
            conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv2bis')
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv3')
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], name='conv3bis')
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], name='conv4')
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], name='conv4bis')
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 4, 1], name='conv4bisbis')

        with tf.name_scope('dnn'):
            conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
            full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 128, name='dnn1')
            full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
            full_layer_2, mf2 = normal_full_layer(full_dropout1, patch_size ** 2 // 128, name='dnn2')
            full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
            full_layer_3, mf3 = normal_full_layer(full_dropout2, patch_size ** 2 // 64, name='dnn3')
            full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout3')
            dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_transpose_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 4], [X_dyn_batsize, patch_size // 8], name='deconv5')  #[height, width, in_channels, output_channels]
            deconv_5bis, m5b = conv2d_transpose_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 8], [X_dyn_batsize, patch_size // 8],name='deconv5bis')  #fixme: strides should be 2
            concat1 = concat([up_2by2(deconv_5bis, name='up1'), conv3bis], name='concat1')  #note: up_2by2 slowing than conv2d_transpose2x2

            deconv_6, m6 = conv2d_transpose_layer(concat1, [conv_size, conv_size, nb_conv * 10, nb_conv * 2], [X_dyn_batsize, patch_size // 4], name='deconv6')
            deconv_6bis, m6b = conv2d_transpose_layer(deconv_6, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], [X_dyn_batsize, patch_size // 4], name='deconv6bis')  #fixme: strides should be 2
            concat2 = concat([up_2by2(deconv_6bis, name='up2'), conv2bis], name='concat2')  #note: up_2by2 slowing than conv2d_transpose2x2

            deconv_7, m7 = conv2d_transpose_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], [X_dyn_batsize, patch_size // 2], name='deconv7')
            deconv_7bis, m7b = conv2d_transpose_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], [X_dyn_batsize, patch_size // 2], name='deconv7bis')  #fixme: strides should be 2
            concat3 = concat([up_2by2(deconv_7bis, name='up3'), conv1bis], name='concat3')  #note: up_2by2 slowing than conv2d_transpose2x2

            deconv_8, m8 = conv2d_transpose_layer(concat3, [conv_size, conv_size, nb_conv * 3, nb_conv], [X_dyn_batsize, patch_size], name='deconv8')
            deconv_8bis, m8b = conv2d_transpose_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], [X_dyn_batsize, patch_size], name='deconv8bis')
            logits, m8bb = conv2d_transpose_layer(deconv_8bis, [conv_size, conv_size, nb_conv, 1], [X_dyn_batsize, patch_size], name='logits')  #fixme: change activation function. 0 everywhere prediction?

    with tf.name_scope('operation'):
        # optimizer/train operation
        mse = loss_fn(inputs['label'], logits, name='loss_fn')
        opt = optimizer(learning_rate, name='optimizeR')

        # program gradients
        grads = opt.compute_gradients(mse)
        grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])

        # train operation
        train_op = opt.apply_gradients(grads, name='train_op')

    with tf.name_scope('metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, inputs['label'], mse, training_type)

    with tf.name_scope('summary'):
        merged = tf.summary.merge([m1, m1b, m2, m2b, m3, m3b, m4, m4b, m4bb, mf1, mf2, mf3,
                                   m5, m5b, m6, m6b, m7, m7b, m8, m8b, m8bb, m_loss, m_acc, grad_sum])  #fixme: withdraw summary of histories for GPU resource reason
    return {
        'y_pred': logits,
        'train_op': train_op,
        'drop': drop_prob,
        'summary': merged,
        'train_or_test': training_type,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }
