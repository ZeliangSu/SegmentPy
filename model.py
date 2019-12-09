import tensorflow as tf
from layers import *
from util import print_nodes_name_shape


def nodes(pipeline,
          placeholders=None,
          model_name='LRCS',
          patch_size=512,
          batch_size=200,
          conv_size=9,
          nb_conv=80,
          activation='relu',
          is_training=False,
          mode='regression'):

    # check entries
    assert isinstance(placeholders, list), 'placeholders should be a list.'
    # get placeholder
    drop_prob, lr, BN_phase = placeholders

    # build model
    logits, list_params = model_dict[model_name](pipeline=pipeline,
                                                 patch_size=patch_size,
                                                 batch_size=batch_size,
                                                 conv_size=conv_size,
                                                 nb_conv=nb_conv,
                                                 drop_prob=drop_prob,
                                                 activation=activation,
                                                 BN_phase=BN_phase,
                                                 reuse=not is_training,
                                                 mode=mode,
                                                 nb_classes=3,
                                                 )

    with tf.device('/cpu:0'):
        with tf.name_scope('Loss'):
            loss = DSC(pipeline['label'], logits, name='loss_fn')
    # loss function
    if is_training:
        with tf.device('/device:GPU:0'):
            with tf.name_scope('operation'):
                # optimizer/train operation
                opt = optimizer(lr, name='optimizeR')

                # program gradients
                grads = opt.compute_gradients(loss)

                # train operation
                train_op = opt.apply_gradients(grads, name='train_op')

        with tf.name_scope('train_metrics'):
            m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, pipeline['label'], loss, is_training)

        with tf.name_scope('summary'):
            grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
            tmp = []
            for layer_param in list_params:
                for k, v in layer_param.items():
                    tmp.append(tf.summary.histogram(k, v))
            m_param = tf.summary.merge(tmp)
            merged = tf.summary.merge([m_param, m_loss, m_acc, grad_sum])
    else:
        with tf.device('/device:GPU:0'):
            with tf.name_scope('operation'):
                train_op = tf.no_op(name='no_op')
        with tf.name_scope('test_metrics'):
            m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, pipeline['label'], loss, is_training)
        with tf.name_scope('summary'):
            tmp = []
            for layer_param in list_params:
                for k, v in layer_param.items():
                    tmp.append(tf.summary.histogram(k, v))
            m_param = tf.summary.merge(tmp)
            merged = tf.summary.merge([m_param, m_loss, m_acc])

    return {
        'y_pred': logits,
        'train_op': train_op,
        'learning_rate': lr,
        'summary': merged,
        'drop': drop_prob,
        'BN_phase': BN_phase,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op
    }


def model_LRCS(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               BN_phase,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3):
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
    #note: Batch Norm automatically applied, can be tuned manually
    with tf.device('/device:GPU:0' if reuse==False else '/device:GPU:1'):
        with tf.name_scope('LRCS'):
            with tf.name_scope('encoder'):
                conv1, _ = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv],
                                        is_train=BN_phase, activation=activation,
                                        name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
                conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                           is_train=BN_phase, activation=activation,
                                           name='conv1bis', reuse=reuse)
                conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1bis, name='maxp1')

                conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                        is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
                conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                           is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
                conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2bis, name='maxp2')

                conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                        is_train=BN_phase,
                                        activation=activation, name='conv3', reuse=reuse)
                conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation=activation, name='conv3bis', reuse=reuse)
                conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3bis, name='maxp3')

                conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                         is_train=BN_phase, activation=activation, name='conv4', reuse=reuse)
                conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                             is_train=BN_phase, activation=activation, name='conv4bis', reuse=reuse)
                conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 8, 1],
                                                 is_train=BN_phase, activation=activation,
                                                 name='conv4bisbis', reuse=reuse)

            with tf.name_scope('dnn'):
                conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
                full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 128, activation=activation,
                                                      is_train=BN_phase, name='dnn1', reuse=reuse)
                full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
                full_layer_2, mf2 = normal_full_layer(full_dropout1, patch_size ** 2 // 128, activation=activation,
                                                      is_train=BN_phase, name='dnn2', reuse=reuse)
                full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
                full_layer_3, mf3 = normal_full_layer(full_dropout2, patch_size ** 2 // 64, activation=activation,
                                                      is_train=BN_phase, name='dnn3', reuse=reuse)
                full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout1')
                dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

            with tf.name_scope('decoder'):
                deconv_5, m5 = conv2d_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 8],
                                           is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)  # [height, width, in_channels, output_channels]
                deconv_5bis, _ = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                              is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)

                up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
                deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                           is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
                deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                              is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

                up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
                deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                           is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
                deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                              is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

                up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
                deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv, nb_conv],
                                           is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
                deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv],
                                              is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
                logits, m8bb = conv2d_layer(deconv_8bis,
                                            [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                            if_BN=False,is_train=BN_phase,
                                            name='logits', reuse=reuse)

        return logits, [m3b, m4bb, mf1, mf2, mf3, m5, m8bb]


def model_Unet(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               BN_phase,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3
               ):
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
    with tf.device('/device:GPU:0' if reuse == False else '/device:GPU:1'):
        with tf.name_scope('Unet'):
            with tf.name_scope('contractor'):
                conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], #[height, width, in_channels, output_channels]
                                        is_train=BN_phase, activation=activation,
                                        name='conv1', reuse=reuse)
                conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                           is_train=BN_phase,
                                           activation=activation, name='conv1bis', reuse=reuse)
                conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

                conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                        is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
                conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                           is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
                conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

                conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                        is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
                conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation=activation, name='conv3bis', reuse=reuse)
                conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

                conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                         is_train=BN_phase, activation=activation, name='conv4', reuse=reuse)
                conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, 8],
                                                 is_train=BN_phase, activation=activation,
                                                 name='conv4bisbis', reuse=reuse)
                conv4_pooling = max_pool_2by2(conv4bis, name='maxp4')

            with tf.name_scope('bottom'):
                conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16],
                                         activation=activation, name='bot5', reuse=reuse)
                conv5bis, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16],
                                             activation=activation, name='bot5bis', reuse=reuse)
                deconv1, m5u = conv2d_transpose_layer(conv5bis, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                                      # fixme: batch_size here might not be automatic while inference (try tf.shape()?)
                                                      [batch_size, patch_size // 8, patch_size // 8, nb_conv * 8],
                                                      stride=2, activation=activation, name='deconv1', reuse=reuse)

            with tf.name_scope('decontractor'):
                concat1 = concat([deconv1, conv4_pooling], name='concat1')
                conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                          activation=activation, name='conv6', reuse=reuse)  #[height, width, in_channels, output_channels]
                conv_6bis, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                              activation=activation, name='conv6bis', reuse=reuse)
                deconv2, m6u = conv2d_transpose_layer(conv_6bis, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size // 4, patch_size //4, nb_conv * 4],
                                                      stride=2, activation=activation,
                                                      name='deconv2', reuse=reuse)

                concat2 = concat([deconv2, conv3bis], name='concat2')
                conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                          activation=activation, name='conv7', reuse=reuse)
                conv_7bis, m7b = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                              activation=activation, name='conv7bis', reuse=reuse)
                deconv3, m7u = conv2d_transpose_layer(conv_7bis, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size // 2, patch_size // 2, nb_conv * 2],
                                                      stride=2, activation=activation,
                                                      name='deconv3', reuse=reuse)

                concat3 = concat([deconv3, conv2bis], name='concat3')
                conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                          activation=activation, name='conv8', reuse=reuse)
                conv_8bis, m8b = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                              activation=activation,  name='conv8bis', reuse=reuse)
                deconv4, m8u = conv2d_transpose_layer(conv_8bis, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size, patch_size, nb_conv],
                                                      stride=2, activation=activation,
                                                      name='deconv4', reuse=reuse)

                concat4 = concat([deconv4, conv1bis], name='concat4')
                deconv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                            activation=activation, name='conv9', reuse=reuse)
                deconv_9bis, m9b = conv2d_layer(deconv_9, [conv_size, conv_size, nb_conv, nb_conv],
                                                activation=activation, name='conv9bis', reuse=reuse)
                logits, m8bb = conv2d_layer(deconv_9bis,
                                            [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                            if_BN=False, is_train=BN_phase, name='logits', reuse=reuse)

        return logits, [m3b, m4b, m5, m5b, m5u, m6, m9b]


def model_xlearn(pipeline,
                 patch_size,
                 batch_size,
                 conv_size,
                 nb_conv,
                 drop_prob,
                 BN_phase,
                 activation='relu',
                 reuse=False,
                 mode='regression',
                 nb_classes=3,
                 ):
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
    with tf.device('/device:GPU:0' if reuse == False else '/device:GPU:1'):
        with tf.name_scope('Xlearn'):

            with tf.name_scope('encoder'):
                conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], #[height, width, in_channels, output_channels]
                                        is_train=BN_phase, activation=activation,
                                        name='conv1', reuse=reuse)#[height, width, in_channels, output_channels]
                conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                           is_train=BN_phase,
                                           activation=activation, name='conv1bis', reuse=reuse)
                conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

                conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                        is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
                conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                           is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
                conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

                conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                        is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
                conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation=activation, name='conv3bis', reuse=reuse)
                conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

                conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                         is_train=BN_phase, activation=activation, name='conv4', reuse=reuse)
                conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, 8],
                                                 is_train=BN_phase, activation=activation,
                                                 name='conv4bisbis', reuse=reuse)
                conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 8, 1],
                                                 is_train=BN_phase, activation=activation,
                                                 name='conv4bisbis', reuse=reuse)

            with tf.name_scope('dnn'):
                conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
                full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 128, activation=activation,
                                                      is_train=BN_phase, name='dnn1', reuse=reuse)
                full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
                full_layer_2, mf2 = normal_full_layer(full_dropout1, patch_size ** 2 // 128, activation=activation,
                                                      is_train=BN_phase, name='dnn2', reuse=reuse)
                full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
                full_layer_3, mf3 = normal_full_layer(full_dropout2, patch_size ** 2 // 64, activation=activation,
                                                      is_train=BN_phase, name='dnn3', reuse=reuse)
                full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout3')
                dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

            with tf.name_scope('decoder'):
                deconv_5, m5 = conv2d_transpose_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 4],
                                                      [batch_size, patch_size // 8, patch_size // 8, nb_conv * 4],
                                                      is_train=BN_phase, name='deconv5',
                                                      activation=activation, reuse=reuse)  #[height, width, in_channels, output_channels]
                deconv_5bis, m5b = conv2d_transpose_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                                          # fixme: batch_size here might not be automatic while inference
                                                          [batch_size, patch_size // 8, patch_size // 8, nb_conv * 8],
                                                          is_train=BN_phase, name='deconv5bis',
                                                          activation=activation, reuse=reuse)
                concat1 = concat([up_2by2(deconv_5bis, name='up1'), conv3bis], name='concat1')

                deconv_6, m6 = conv2d_transpose_layer(concat1, [conv_size, conv_size, nb_conv * 10, nb_conv * 2],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size // 4, patch_size // 4, nb_conv * 2],
                                                      is_train=BN_phase, name='deconv6',
                                                      activation=activation, reuse=reuse)
                deconv_6bis, m6b = conv2d_transpose_layer(deconv_6, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                                          # fixme: batch_size here might not be automatic while inference
                                                          [batch_size, patch_size // 4, patch_size // 4, nb_conv * 2],
                                                          is_train=BN_phase, name='deconv6bis',
                                                          activation=activation, reuse=reuse)
                concat2 = concat([up_2by2(deconv_6bis, name='up2'), conv2bis], name='concat2')

                deconv_7, m7 = conv2d_transpose_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size // 2, patch_size // 2, nb_conv * 2],
                                                       is_train=BN_phase, name='deconv7',
                                                      activation=activation, reuse=reuse)
                deconv_7bis, m7b = conv2d_transpose_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                                          # fixme: batch_size here might not be automatic while inference
                                                          [batch_size, patch_size // 2, patch_size //2, nb_conv * 2],
                                                          is_train=BN_phase, name='deconv7bis',
                                                          activation=activation, reuse=reuse)
                concat3 = concat([up_2by2(deconv_7bis, name='up3'), conv1bis], name='concat3')

                deconv_8, m8 = conv2d_transpose_layer(concat3, [conv_size, conv_size, nb_conv * 3, nb_conv],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size, patch_size, nb_conv],
                                                      is_train=BN_phase, name='deconv8',
                                                      activation=activation, reuse=reuse)
                deconv_8bis, m8b = conv2d_transpose_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv],
                                                          # fixme: batch_size here might not be automatic while inference
                                                          [batch_size, patch_size, patch_size, nb_conv],
                                                          is_train=BN_phase, name='deconv8bis',
                                                          activation=activation, reuse=reuse)

                logits, m8bb = conv2d_layer(deconv_8bis,
                                            [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                            # fixme: batch_size here might not be automatic while inference
                                            [batch_size, patch_size, patch_size, 1],
                                            if_BN=False, is_train=BN_phase,
                                            name='logits', reuse=reuse)

        return logits, [m3b, m4bb, mf1, mf2, mf3, m5, m8b]


# ind: index skip connexion, img: image skip connexion, c:conv, d:dense, t:transpose, u:up
def ind_3c_3b_3c(pipeline,
                 patch_size,
                 batch_size,
                 conv_size,
                 nb_conv,
                 drop_prob,
                 BN_phase,
                 activation='relu',
                 reuse=False,
                 second_device=None,
                 mode='regression',
                 nb_classes=3
                 ):
    if not second_device:
        second_device = '/cpu:0'
    with tf.device('/device:GPU:0' if reuse == False else second_device):
        with tf.name_scope('ind_3c_3b_3c'):
            with tf.name_scope('encoder'):
                conv4, m4 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv],
                                         is_train=BN_phase, activation=activation, name='conv4', reuse=reuse)
                conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                             is_train=BN_phase, activation=activation, name='conv4bis', reuse=reuse)
                conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv, 1],
                                                 is_train=BN_phase, activation=activation,
                                                 name='conv4bisbis', reuse=reuse)

            with tf.name_scope('dnn'):
                conv4_flat = reshape(conv4bisbis, [-1, patch_size], name='flatten')  #note patch_size ** 2?
                full_layer_1, mf1 = normal_full_layer(conv4_flat, 1024, activation=activation,
                                                      is_train=BN_phase, name='dnn1', reuse=reuse)
                full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
                full_layer_2, mf2 = normal_full_layer(full_dropout1, 512, activation=activation,
                                                      is_train=BN_phase, name='dnn2', reuse=reuse)
                full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
                full_layer_3, mf3 = normal_full_layer(full_dropout2, 64, activation=activation,
                                                      is_train=BN_phase, name='dnn3', reuse=reuse)
                full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout1')
                dnn_reshape = reshape(full_dropout3, [-1, patch_size, patch_size, 1], name='reshape')

            with tf.name_scope('decoder'):
                deconv_5, m5 = conv2d_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv],
                                            is_train=BN_phase, activation=activation, name='deconv5',
                                            reuse=reuse)  # [height, width, in_channels, output_channels]
                deconv_5bis, m5b = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv, nb_conv],
                                              is_train=BN_phase, activation=activation,
                                              name='deconv5bis', reuse=reuse)
                logits, m5bb = conv2d_layer(deconv_5bis, [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                              is_train=BN_phase, activation=activation, if_BN=False,
                                              name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, [m4, m4b, m4bb, mf1, mf2, mf3, m5, m5b, m5bb]

model_dict = {
    'LRCS': model_LRCS,
    'Xlearn': model_xlearn,
    'Unet': model_Unet,
    'ind_3c_3b_3c': ind_3c_3b_3c,
}