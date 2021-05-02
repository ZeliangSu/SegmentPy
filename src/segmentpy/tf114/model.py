import tensorflow as tf
from layers import *
from util import print_nodes_name_shape


def regression_nodes(pipeline,
                     placeholders=None,
                     model_name='LRCS',
                     patch_size=512,
                     batch_size=200,
                     conv_size=9,
                     nb_conv=80,
                     activation='relu',
                     batch_norm=True,
                     is_training=False,
                     nb_classes=3
                     ):

    # todo: correct
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
                                                 mode='regression',
                                                 nb_classes=nb_classes,
                                                 )

    with tf.device('/cpu:0'):
        with tf.name_scope('Loss'):
            loss = MSE(pipeline['label'], logits, name='loss_fn')
    # loss function
    if is_training:
        with tf.name_scope('operation'):
            # optimizer/train operation
            opt = optimizer(lr, name='optimizeR')

            # program gradients
            grads = opt.compute_gradients(loss)

            # train operation
            train_op = opt.apply_gradients(grads, name='train_op')

        with tf.name_scope('train_metrics'):
            m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, pipeline['label'], loss, is_training, mode='regression')

        with tf.name_scope('summary'):
            grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
            tmp = []
            for layer_param in list_params:
                for k, v in layer_param.items():
                    tmp.append(tf.summary.histogram(k, v))
            m_param = tf.summary.merge(tmp)
            merged = tf.summary.merge([m_param, m_loss, m_acc, grad_sum])
    else:
        with tf.name_scope('operation'):
            train_op = tf.no_op(name='no_op')
        with tf.name_scope('test_metrics'):
            m_loss, loss_up_op, m_acc, acc_up_op = metrics(logits, pipeline['label'], loss, is_training, mode='regression')
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


def classification_nodes(pipeline,
                         placeholders=None,
                         model_name='LRCS',
                         patch_size=512,
                         batch_size=200,
                         conv_size=9,
                         nb_conv=80,
                         activation='relu',
                         batch_norm=True,
                         loss_option='cross_entropy',
                         is_training=False,
                         grad_view=False,
                         nb_classes=3
                         ):

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
                                                 if_BN=batch_norm,
                                                 BN_phase=BN_phase,
                                                 reuse=not is_training,
                                                 mode='classification',
                                                 nb_classes=nb_classes,  # todo: automatize here
                                                 )

    # logits shape [B, H, W, nb_class]
    with tf.name_scope('Loss'):
        if loss_option == 'DSC':
            softmax = customized_softmax(logits)
            loss = DSC(pipeline['label'], softmax, name='loss_fn')
        elif loss_option == 'cross_entropy':
            softmax = customized_softmax(logits)
            loss = Cross_Entropy(pipeline['label'], softmax, name='CE')
        else:
            raise NotImplementedError('Cannot find the loss option')

    # gradients
    if is_training:
        with tf.name_scope('operation'):
            # optimizer/train operation
            opt = optimizer(lr, name='optimizeR')

            # program gradients
            grads = opt.compute_gradients(loss)

            # train operation
            train_op = opt.apply_gradients(grads, name='train_op')

        with tf.name_scope('train_metrics'):
            m_loss, loss_up_op, m_acc, acc_up_op, lss, acc = metrics(softmax,  #[B, W, H, 3]
                                                           pipeline['label'],  #[B, W, H, 3]
                                                           loss,
                                                           is_training)

        with tf.name_scope('summary'):
            tmp = []
            for layer_param in list_params:
                for k, v in layer_param.items():
                    tmp.append(tf.summary.histogram(k, v))
            if len(tmp) > 0:
                m_param = tf.summary.merge(tmp)
                merged = tf.summary.merge([m_param, m_loss, m_acc])
            else:
                merged = tf.summary.merge([m_loss, m_acc])

            if grad_view:
                grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
                merged = tf.summary.merge([merged, grad_sum])

    else:
        with tf.name_scope('operation'):
            train_op = tf.no_op(name='no_op')
        with tf.name_scope('test_metrics'):
            m_loss, loss_up_op, m_acc, acc_up_op, lss, acc = metrics(softmax, pipeline['label'], loss, is_training)
        with tf.name_scope('summary'):
            tmp = []
            for layer_param in list_params:
                for k, v in layer_param.items():
                    tmp.append(tf.summary.histogram(k, v))
            if len(tmp) > 0:
                m_param = tf.summary.merge(tmp)
                merged = tf.summary.merge([m_param, m_loss, m_acc])
            else:
                merged = tf.summary.merge([m_loss, m_acc])

    return {
        'y_pred': logits,
        'train_op': train_op,
        'learning_rate': lr,
        'summary': merged,
        'drop': drop_prob,
        'BN_phase': BN_phase,
        'loss_update_op': loss_up_op,
        'acc_update_op': acc_up_op,
        'val_lss': lss,
        'val_acc': acc,
    }


def model_LRCS(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):
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
    with tf.name_scope('LRCS'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation,
                                       name='conv1bis', reuse=reuse)
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                    if_BN=if_BN, is_train=BN_phase,
                                    activation=activation, name='conv3', reuse=reuse)
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase,
                                     activation=activation, name='conv4', reuse=reuse)
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv4bis', reuse=reuse)
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 8, 1],
                                             if_BN=if_BN, is_train=BN_phase,
                                             activation=activation, name='conv4bisbis', reuse=reuse)

        with tf.name_scope('dnn'):
            conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
            full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 128, activation=activation,  # OOM: //128 --> //512
                                                  if_BN=if_BN, is_train=BN_phase, name='dnn1', reuse=reuse)
            full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
            full_layer_2, mf2 = normal_full_layer(full_dropout1, patch_size ** 2 // 128, activation=activation,  # OOM: //128 --> //512
                                                  if_BN=if_BN, is_train=BN_phase, name='dnn2', reuse=reuse)
            full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
            full_layer_3, mf3 = normal_full_layer(full_dropout2, patch_size ** 2 // 64, activation=activation,  # OOM: //64 --> //512
                                                  if_BN=if_BN, is_train=BN_phase, name='dnn3', reuse=reuse)
            full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout1')
            dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 8], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)  # [height, width, in_channels, output_channels]
            deconv_5bis, _ = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_improved(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS2'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation,
                                       name='conv1bis', reuse=reuse)
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                    if_BN=if_BN, is_train=BN_phase,
                                    activation=activation, name='conv3', reuse=reuse)
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase,
                                     activation=activation, name='conv4', reuse=reuse)
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv4bis', reuse=reuse)
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 8, nb_classes],
                                             if_BN=if_BN, is_train=BN_phase,
                                             activation=activation, name='conv4bisbis', reuse=reuse)

        with tf.name_scope('dnn'):
            conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64 * nb_classes], name='flatten')
            full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 1024 * nb_classes, activation=activation,
                                                  if_BN=if_BN, is_train=BN_phase, name='dnn1', reuse=reuse)
            full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')

            # add a second layer can reduce NxN --> 2xMxN
            full_layer_2, mf2 = normal_full_layer(full_dropout1, nb_classes, activation=activation,  # note: shoudnt do nb_classes * batch
                                                  if_BN=if_BN, is_train=BN_phase, name='dnn2', reuse=reuse)
            full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')

            full_layer_3, mf3 = normal_full_layer(full_dropout2, patch_size ** 2 // 64 * nb_classes,
                                                  activation=activation,
                                                  if_BN=if_BN, is_train=BN_phase, name='dnn3',
                                                  reuse=reuse)
            full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout3')
            dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, nb_classes], name='reshape')

        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_layer(dnn_reshape, [conv_size, conv_size, nb_classes, nb_conv * 8], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)  # [height, width, in_channels, output_channels]
            deconv_5bis, _ = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_constant(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    #note: Batch Norm automatically applied, can be tuned manually
    with tf.name_scope('LRCS3'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation,
                                       name='conv1bis', reuse=reuse)
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                    if_BN=if_BN, is_train=BN_phase,
                                    activation=activation, name='conv3', reuse=reuse)
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase,
                                     activation=activation, name='conv4', reuse=reuse)
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv4bis', reuse=reuse)
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 8, nb_classes],
                                             if_BN=if_BN, is_train=BN_phase,
                                             activation=activation, name='conv4bisbis', reuse=reuse)

        with tf.name_scope('dnn'):
            dnn_reshape = constant_layer(conv4bisbis, constant=1.0)

        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_layer(dnn_reshape, [conv_size, conv_size, nb_classes, nb_conv * 8], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)  # [height, width, in_channels, output_channels]
            deconv_5bis, m5b = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, [m3b, m4bb, m5, m5b, m8bb]


def model_LRCS_shallow(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS4'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[1, 1, 1, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3, name='maxp3')

            conv4bisbis, m4bb = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_classes],
                                             is_train=BN_phase, activation=activation,if_BN=False,
                                             name='conv4bisbis', reuse=reuse)

        with tf.name_scope('dnn'):
            dnn_reshape = constant_layer(conv4bisbis, constant=1.0, name='constant')

        with tf.name_scope('decoder'):
            deconv_5bis, m5b = conv2d_layer(dnn_reshape, [conv_size, conv_size, nb_classes, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
    print_nodes_name_shape(tf.get_default_graph())
    return logits, []


def model_LRCS_simple(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS5'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[1, 1, 1, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation,
                                     name='conv1', reuse=reuse, if_BN=False)  # [height, width, in_channels, output_channels]
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse, if_BN=False)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2, name='maxp2')

        with tf.name_scope('connexion'):
            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)

        with tf.name_scope('decoder'):
            up1 = up_2by2_ind(conv3, ind2, name='up2')
            concat1 = concat([up1, conv1_pooling])
            deconv_7, m7 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 2 + nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            # concat2 = concat([up2, pipeline['img']])
            deconv_8, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, [m1, m3, m7, m8bb]


def model_LRCS_purConv(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS6'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv * 2], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1b, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation,
                                       name='conv1b', reuse=reuse)

            conv1bb, m1bb = conv2d_layer(conv1b, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation,
                                       name='conv1bb', reuse=reuse)

            logits, m1bbb = conv2d_layer(conv1bb, shape=[conv_size, conv_size, nb_conv * 2, nb_classes], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation,
                                       name='conv1bbb', reuse=reuse)

        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_LeCun(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS7'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1b, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1bis', reuse=reuse)
            conv1bb, m1bb = conv2d_layer(conv1b, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1bisbis', reuse=reuse)
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1bb, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2b, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2b, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3b, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3b, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation=activation, if_BN=False,
                                             name='conv4', reuse=reuse)
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation='sigmoid', if_BN=False,
                                             name='conv4bis', reuse=reuse)

        with tf.name_scope('decoder'):
            deconv5, m5 = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation=activation, if_BN=if_BN,
                                             name='conv5', reuse=reuse)
            deconv_5bis, m5b = conv2d_layer(deconv5, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_Weka(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS8'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 10, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=True,
                                     name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=True,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], if_BN=True,
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3, name='maxp3')

            conv4bisbis, m4bb = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_classes],
                                             is_train=BN_phase, activation='sigmoid', if_BN=True,
                                             name='conv4bisbis', reuse=reuse)

        with tf.name_scope('decoder'):
            deconv_5bis, m5b = conv2d_layer(conv4bisbis, [conv_size, conv_size, nb_classes, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_weka_constant(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS9'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 10, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3, name='maxp3')

            conv4bisbis, m4bb = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_classes],
                                             is_train=BN_phase, activation=activation,if_BN=False,
                                             name='conv4bisbis', reuse=reuse)

        with tf.name_scope('dnn'):
            dnn_reshape = constant_layer(conv4bisbis, constant=1.0, name='constant')

        with tf.name_scope('decoder'):
            deconv_5bis, m5b = conv2d_layer(dnn_reshape, [conv_size, conv_size, nb_classes, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 6], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 6, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_lecun_thinner_weka_encoder(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS10'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 10, 20],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, 20, 40], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, 40, 80], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3, name='maxp3')

            conv4bisbis, m4bb = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, 80, 10],
                                             is_train=BN_phase, activation='sigmoid', if_BN=False,
                                             name='conv4bisbis', reuse=reuse)

        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_layer(conv4bisbis, [conv_size, conv_size, 10, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)

            deconv_5bis, m5b = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, 80],
                                            if_BN=if_BN,
                                            is_train=BN_phase, activation=activation, name='deconv5bis',
                                            reuse=reuse)

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, 80, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, 40], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, 40, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, 20], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, 20, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_lecun_thinner_encoder(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS11'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation='sigmoid', if_BN=False,
                                             name='conv4', reuse=reuse)
        # note: wider connexion for the bottom layers
        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_layer(conv4, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)

            deconv_5bis, m5b = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                            if_BN=if_BN,
                                            is_train=BN_phase, activation=activation, name='deconv5bis',
                                            reuse=reuse)

            up1 = up_2by2_ind(deconv_5bis, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_mix_skipconnect(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS12'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation='sigmoid', if_BN=False,
                                             name='conv4', reuse=reuse)
        # note: wider connexion for the bottom layers
        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_layer(conv4, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)

            deconv_5bis, m5b = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                            if_BN=if_BN,
                                            is_train=BN_phase, activation=activation, name='deconv5bis',
                                            reuse=reuse)

            concat1 = concat([up_2by2_ind(deconv_5bis, ind3, name='up1'), conv3], name='concat1')
            deconv_6, _ = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            concat2 = concat([up_2by2_ind(deconv_6bis, ind2, name='up2'), conv2], name='concat2')
            deconv_7, _ = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            concat3 = concat([up_2by2_ind(deconv_7bis, ind1, name='up3'), conv1], name='concat3')
            deconv_8, _ = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_dropout_on_conv(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS13'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            drop1 = dropout(conv1, drop_prob, name='do1')
            conv1_pooling, ind1 = max_pool_2by2_with_arg(drop1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            drop2 = dropout(conv2, drop_prob, name='do2')
            conv2_pooling, ind2 = max_pool_2by2_with_arg(drop2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)

            drop3 = dropout(conv3, drop_prob, name='do3')
            conv3_pooling, ind3 = max_pool_2by2_with_arg(drop3, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation='sigmoid', if_BN=False,
                                             name='conv4', reuse=reuse)
            drop4 = dropout(conv4, drop_prob, name='do4')

        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_layer(drop4, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)

            deconv_5bis, m5b = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                            if_BN=if_BN,
                                            is_train=BN_phase, activation=activation, name='deconv5bis',
                                            reuse=reuse)
            drop5 = dropout(deconv_5bis, drop_prob, name='do5')


            up1 = up_2by2_ind(drop5, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)
            drop6 = dropout(deconv_6bis, drop_prob, name='do6')

            up2 = up_2by2_ind(drop6, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)
            drop7 = dropout(deconv_7bis, drop_prob, name='do7')

            up3 = up_2by2_ind(drop7, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_LRCS_full_FCLs(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):
    with tf.name_scope('LRCS14'):
        with tf.name_scope('encoder'):
            pass
        with tf.name_scope('decoder'):
            pass


def model_LRCS_deeper_with_dropout_on_conv(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('LRCS15'):
        with tf.name_scope('encoder'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            drop1 = dropout(conv1, drop_prob, name='do1')
            conv1b, m1 = conv2d_layer(drop1, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                     # [height, width, in_channels, output_channels]
                                     is_train=BN_phase, activation=activation, if_BN=False,
                                     name='conv1b', reuse=reuse)  # [height, width, in_channels, output_channels]
            drop1b = dropout(conv1b, drop_prob, name='do1b')
            conv1_pooling, ind1 = max_pool_2by2_with_arg(drop1b, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            drop2 = dropout(conv2, drop_prob, name='do2')
            conv2b, m2 = conv2d_layer(drop2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv2b', reuse=reuse)
            drop2b = dropout(conv2b, drop_prob, name='do2b')
            conv2_pooling, ind2 = max_pool_2by2_with_arg(drop2b, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            drop3 = dropout(conv3, drop_prob, name='do3')
            conv3b, m3 = conv2d_layer(drop3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=False,
                                    is_train=BN_phase, activation=activation, name='conv3b', reuse=reuse)
            drop3b = dropout(conv3b, drop_prob, name='do3b')
            conv3_pooling, ind3 = max_pool_2by2_with_arg(drop3b, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation=activation, if_BN=False,
                                             name='conv4', reuse=reuse)
            drop4 = dropout(conv4, drop_prob, name='do4')
            conv4b, m4 = conv2d_layer(drop4, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                             is_train=BN_phase, activation='sigmoid', if_BN=False,
                                             name='conv4b', reuse=reuse)
            drop4 = dropout(conv4b, drop_prob, name='do4b')

        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_layer(drop4, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)
            drop5 = dropout(deconv_5, drop_prob, name='do5')
            deconv_5bis, m5b = conv2d_layer(drop5, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                            if_BN=if_BN,
                                            is_train=BN_phase, activation=activation, name='deconv5bis',
                                            reuse=reuse)
            drop5b = dropout(deconv_5bis, drop_prob, name='do5b')

            up1 = up_2by2_ind(drop5b, ind3, name='up1')
            deconv_6, _ = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            drop6 = dropout(deconv_6, drop_prob, name='do6')
            deconv_6bis, _ = conv2d_layer(drop6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)
            drop6b = dropout(deconv_6bis, drop_prob, name='do6b')

            up2 = up_2by2_ind(drop6b, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            drop7 = dropout(deconv_7, drop_prob, name='do7')
            deconv_7bis, _ = conv2d_layer(drop7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)
            drop7b = dropout(deconv_7bis, drop_prob, name='do7b')

            up3 = up_2by2_ind(drop7b, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            drop8 = dropout(deconv_8, drop_prob, name='do8')
            deconv_8bis, _ = conv2d_layer(drop8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            drop8b = dropout(deconv_8bis, drop_prob, name='do8b')
            logits, m8bb = conv2d_layer(drop8b,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Segnet_like(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('Segnet'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation,
                                       name='conv1bis', reuse=reuse)
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2bis, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2bis, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                    if_BN=if_BN, is_train=BN_phase,
                                    activation=activation, name='conv3', reuse=reuse)
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase,
                                     activation=activation, name='conv4', reuse=reuse)
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv4bis', reuse=reuse)
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 8, 1],
                                             if_BN=if_BN, is_train=BN_phase,
                                             activation=activation, name='conv4bisbis', reuse=reuse)
            conv4_pooling, ind4 = max_pool_2by2_with_arg(conv4bisbis, name='maxp4')

        with tf.name_scope('decoder'):
            up0 = up_2by2_ind(conv4_pooling, ind4, name='up0')
            deconv_5, m5 = conv2d_layer(up0, [conv_size, conv_size, 1, nb_conv * 8], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv5', reuse=reuse)  # [height, width, in_channels, output_channels]
            deconv_5bis, _ = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)
            deconv_5bisbis, _ = conv2d_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv5bis', reuse=reuse)

            up1 = up_2by2_ind(deconv_5bisbis, ind3, name='up1')
            deconv_6, m6 = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv6', reuse=reuse)
            deconv_6bis, _ = conv2d_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv6bis', reuse=reuse)

            up2 = up_2by2_ind(deconv_6bis, ind2, name='up2')
            deconv_7, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv7', reuse=reuse)
            deconv_7bis, _ = conv2d_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv7bis', reuse=reuse)

            up3 = up_2by2_ind(deconv_7bis, ind1, name='up3')
            deconv_8, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='deconv8', reuse=reuse)
            deconv_8bis, _ = conv2d_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='deconv8bis', reuse=reuse)
            logits, m8bb = conv2d_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Segnet_improved(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('Segnet2'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation,
                                       name='conv1bis', reuse=reuse)
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2b, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2b, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                    if_BN=if_BN, is_train=BN_phase,
                                    activation=activation, name='conv3', reuse=reuse)
            conv3b, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3b, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase,
                                     activation=activation, name='conv4', reuse=reuse)
            conv4b, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv4bis', reuse=reuse)
            conv4_pooling, ind4 = max_pool_2by2_with_arg(conv4b, name='maxp4')

        with tf.name_scope('connexion'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5', reuse=reuse)
            conv5b, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16],
                                         if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5bis', reuse=reuse)
            conv5bb, m5u = conv2d_layer(conv5b, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                        if_BN=if_BN, is_train=BN_phase,
                                        activation=activation, name='bot5bb', reuse=reuse)

        with tf.name_scope('decoder'):
            up0 = up_2by2_ind(conv5bb, ind4, name='up0')
            conv6, m6 = conv2d_layer(up0, [conv_size, conv_size, nb_conv * 8, nb_conv * 8], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv6', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv6b, _ = conv2d_layer(conv6, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv6bis', reuse=reuse)

            up1 = up_2by2_ind(conv6b, ind3, name='up1')
            conv7, m7 = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv7', reuse=reuse)
            conv7b, _ = conv2d_layer(conv7, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv7bis', reuse=reuse)

            up2 = up_2by2_ind(conv7b, ind2, name='up2')
            conv8, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv8', reuse=reuse)
            conv8b, _ = conv2d_layer(conv8, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv8bis', reuse=reuse)

            up3 = up_2by2_ind(conv8b, ind1, name='up3')
            conv9, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv9', reuse=reuse)
            conv9b, _ = conv2d_layer(conv9, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv9bis', reuse=reuse)
            logits, m9bb = conv2d_layer(conv9b,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Segnet_constant(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('Segnet3'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv1bis, _ = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation,
                                       name='conv1bis', reuse=reuse)
            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1bis, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2b, _ = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2b, name='maxp2')

            conv3, _ = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                    if_BN=if_BN, is_train=BN_phase,
                                    activation=activation, name='conv3', reuse=reuse)
            conv3b, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3b, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase,
                                     activation=activation, name='conv4', reuse=reuse)
            conv4b, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                         if_BN=if_BN, is_train=BN_phase,
                                         activation=activation, name='conv4bis', reuse=reuse)
            conv4_pooling, ind4 = max_pool_2by2_with_arg(conv4b, name='maxp4')

        with tf.name_scope('connexion'):
            connex = constant_layer(conv4_pooling, constant=1.0, name='constant')

        with tf.name_scope('decoder'):
            up0 = up_2by2_ind(connex, ind4, name='up0')
            conv6, m6 = conv2d_layer(up0, [conv_size, conv_size, nb_conv * 8, nb_conv * 8], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv6', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv6b, _ = conv2d_layer(conv6, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv6bis', reuse=reuse)

            up1 = up_2by2_ind(conv6b, ind3, name='up1')
            conv7, m7 = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv7', reuse=reuse)
            conv7b, _ = conv2d_layer(conv7, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv7bis', reuse=reuse)

            up2 = up_2by2_ind(conv7b, ind2, name='up2')
            conv8, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv8', reuse=reuse)
            conv8b, _ = conv2d_layer(conv8, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv8bis', reuse=reuse)

            up3 = up_2by2_ind(conv8b, ind1, name='up3')
            conv9, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv9', reuse=reuse)
            conv9b, _ = conv2d_layer(conv9, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv9bis', reuse=reuse)
            logits, m9bb = conv2d_layer(conv9b,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Segnet_shallow(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('Segnet4'):
        with tf.name_scope('encoder'):
            conv1, _ = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)  # [height, width, in_channels, output_channels]

            conv1_pooling, ind1 = max_pool_2by2_with_arg(conv1, name='maxp1')

            conv2, _ = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2], if_BN=if_BN,
                                    is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling, ind2 = max_pool_2by2_with_arg(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                    if_BN=if_BN, is_train=BN_phase,
                                    activation=activation, name='conv3', reuse=reuse)
            conv3_pooling, ind3 = max_pool_2by2_with_arg(conv3, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_classes],
                                     if_BN=if_BN, is_train=BN_phase,
                                     activation=activation, name='conv4', reuse=reuse)
            conv4_pooling, ind4 = max_pool_2by2_with_arg(conv4, name='maxp4')

        with tf.name_scope('decoder'):
            up0 = up_2by2_ind(conv4_pooling, ind4, name='up0')
            conv6, m6 = conv2d_layer(up0, [conv_size, conv_size, nb_classes, nb_conv * 8], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv6', reuse=reuse)  # [height, width, in_channels, output_channels]
            conv6b, _ = conv2d_layer(conv6, [conv_size, conv_size, nb_conv * 8, nb_conv * 4], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv6bis', reuse=reuse)

            up1 = up_2by2_ind(conv6b, ind3, name='up1')
            conv7, m7 = conv2d_layer(up1, [conv_size, conv_size, nb_conv * 4, nb_conv * 4], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv7', reuse=reuse)
            conv7b, _ = conv2d_layer(conv7, [conv_size, conv_size, nb_conv * 4, nb_conv * 2], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv7bis', reuse=reuse)

            up2 = up_2by2_ind(conv7b, ind2, name='up2')
            conv8, _ = conv2d_layer(up2, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv8', reuse=reuse)
            conv8b, _ = conv2d_layer(conv8, [conv_size, conv_size, nb_conv * 2, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv8bis', reuse=reuse)

            up3 = up_2by2_ind(conv8b, ind1, name='up3')
            conv9, _ = conv2d_layer(up3, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                       is_train=BN_phase, activation=activation, name='conv9', reuse=reuse)
            conv9b, _ = conv2d_layer(conv9, [conv_size, conv_size, nb_conv, nb_conv], if_BN=if_BN,
                                          is_train=BN_phase, activation=activation, name='conv9bis', reuse=reuse)
            logits, m9bb = conv2d_layer(conv9b,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False,is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Unet(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('Unet'):
        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], #[height, width, in_channels, output_channels]
                                     if_BN=if_BN, is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)
            conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                         if_BN=if_BN, is_train=BN_phase,
                                       activation=activation, name='conv1bis', reuse=reuse)
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                         if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv4', reuse=reuse)
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                             is_train=BN_phase, activation=activation,
                                             name='conv4bisbis', reuse=reuse)
            conv4_pooling = max_pool_2by2(conv4bis, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5', reuse=reuse)
            conv5bis, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16],
                                         if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5bis', reuse=reuse)
            deconv1, m5u = conv2d_transpose_layer(conv5bis, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv1', reuse=reuse)

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4bis], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv6', reuse=reuse)
            conv_6bis, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                          if_BN=if_BN, is_train=BN_phase,
                                          activation=activation, name='conv6bis', reuse=reuse)
            deconv2, m6u = conv2d_transpose_layer(conv_6bis, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation,
                                                  name='deconv2', reuse=reuse)

            concat2 = concat([deconv2, conv3bis], name='concat2')
            conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv7', reuse=reuse)
            conv_7bis, m7b = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                          if_BN=if_BN, is_train=BN_phase,
                                          activation=activation, name='conv7bis', reuse=reuse)
            deconv3, m7u = conv2d_transpose_layer(conv_7bis, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv3', reuse=reuse)

            concat3 = concat([deconv3, conv2bis], name='concat3')
            conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv8', reuse=reuse)
            conv_8bis, m8b = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                          if_BN=if_BN, is_train=BN_phase,
                                          activation=activation,  name='conv8bis', reuse=reuse)
            deconv4, m8u = conv2d_transpose_layer(conv_8bis, shape=[conv_size, conv_size, nb_conv * 2, nb_conv],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv4', reuse=reuse)

            concat4 = concat([deconv4, conv1bis], name='concat4')
            deconv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                        if_BN=if_BN, is_train=BN_phase,
                                        activation=activation, name='conv9', reuse=reuse)
            deconv_9bis, m9b = conv2d_layer(deconv_9, [conv_size, conv_size, nb_conv, nb_conv],
                                            if_BN=if_BN, is_train=BN_phase,
                                            activation=activation, name='conv9bis', reuse=reuse)
            logits, m9bb = conv2d_layer(deconv_9bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, is_train=BN_phase, name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Unet_shallow(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('Unet2'):
        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], #[height, width, in_channels, output_channels]
                                     if_BN=False, is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)
            conv1_pooling = max_pool_2by2(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                     if_BN=False, is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling = max_pool_2by2(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     if_BN=False, is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3_pooling = max_pool_2by2(conv3, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=False, is_train=BN_phase, activation=activation, name='conv4', reuse=reuse)
            conv4_pooling = max_pool_2by2(conv4, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5', reuse=reuse)
            conv5bis, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16],
                                         if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5bis', reuse=reuse)
            deconv1, m5u = conv2d_transpose_layer(conv5bis, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv1', reuse=reuse)

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv6', reuse=reuse)
            conv_6bis, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                          if_BN=if_BN, is_train=BN_phase,
                                          activation=activation, name='conv6bis', reuse=reuse)
            deconv2, m6u = conv2d_transpose_layer(conv_6bis, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation,
                                                  name='deconv2', reuse=reuse)

            concat2 = concat([deconv2, conv3], name='concat2')
            conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv7', reuse=reuse)
            conv_7bis, m7b = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                          if_BN=if_BN, is_train=BN_phase,
                                          activation=activation, name='conv7bis', reuse=reuse)
            deconv3, m7u = conv2d_transpose_layer(conv_7bis, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv3', reuse=reuse)

            concat3 = concat([deconv3, conv2], name='concat3')
            conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv8', reuse=reuse)
            conv_8bis, m8b = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                          if_BN=if_BN, is_train=BN_phase,
                                          activation=activation,  name='conv8bis', reuse=reuse)
            deconv4, m8u = conv2d_transpose_layer(conv_8bis, shape=[conv_size, conv_size, nb_conv * 2, nb_conv],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv4', reuse=reuse)

            concat4 = concat([deconv4, conv1], name='concat4')
            deconv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                        if_BN=if_BN, is_train=BN_phase,
                                        activation=activation, name='conv9', reuse=reuse)
            deconv_9bis, m9b = conv2d_layer(deconv_9, [conv_size, conv_size, nb_conv, nb_conv],
                                            if_BN=if_BN, is_train=BN_phase,
                                            activation=activation, name='conv9bis', reuse=reuse)
            logits, m9bb = conv2d_layer(deconv_9bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, is_train=BN_phase, name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Unet_weka(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('Unet4'):
        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 10, nb_conv], #[height, width, in_channels, output_channels]
                                     if_BN=if_BN, is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)
            conv1_pooling = max_pool_2by2(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling = max_pool_2by2(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3_pooling = max_pool_2by2(conv3, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv4', reuse=reuse)
            conv4_pooling = max_pool_2by2(conv4, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5', reuse=reuse)
            deconv1, m5u = up_2by2(conv5, name='up1')

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv6', reuse=reuse)  #[height, width, in_channels, output_channels]
            deconv2, m6u = up_2by2(conv_6, name='up2')

            concat2 = concat([deconv2, conv3], name='concat2')
            conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv7', reuse=reuse)
            deconv3, m6u = up_2by2(conv_7, name='up2')

            concat3 = concat([deconv3, conv2], name='concat3')
            conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv8', reuse=reuse)
            deconv4, m8u = conv2d_transpose_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                                  # fixme: batch_size here might not be automatic while inference
                                                  [batch_size, patch_size, patch_size, nb_conv],
                                                  if_BN=if_BN, is_train=BN_phase,
                                                  stride=2, activation=activation,
                                                  name='deconv4', reuse=reuse)

            concat4 = concat([deconv4, conv1], name='concat4')
            deconv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                        if_BN=if_BN, is_train=BN_phase,
                                        activation=activation, name='conv9', reuse=reuse)
            logits, m9bb = conv2d_layer(deconv_9,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, is_train=BN_phase, name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Unet_upsample(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('Unet4'):
        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], #[height, width, in_channels, output_channels]
                                     if_BN=if_BN, is_train=BN_phase, activation=activation,
                                    name='conv1', reuse=reuse)
            conv1_pooling = max_pool_2by2(conv1, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv2', reuse=reuse)
            conv2_pooling = max_pool_2by2(conv2, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv3', reuse=reuse)
            conv3_pooling = max_pool_2by2(conv3, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='conv4', reuse=reuse)
            conv4_pooling = max_pool_2by2(conv4, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5', reuse=reuse)
            conv5b, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5bis', reuse=reuse)
            deconv1, m5u = up_2by2(conv5b, name='up1')

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv6', reuse=reuse)
            conv_6b, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv6bis', reuse=reuse)  #[height, width, in_channels, output_channels]
            deconv2, m6u = up_2by2(conv_6b, name='up2')

            concat2 = concat([deconv2, conv3], name='concat2')
            conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv7', reuse=reuse)
            conv_7b, m7b = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv7bis', reuse=reuse)
            deconv3, m6u = up_2by2(conv_7b, name='up3')

            concat3 = concat([deconv3, conv2], name='concat3')
            conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv8', reuse=reuse)
            conv_8b, m8b = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 1],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv8bis', reuse=reuse)
            deconv4, m6u = up_2by2(conv_8b, name='up4')

            concat4 = concat([deconv4, conv1], name='concat4')
            conv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                        if_BN=if_BN, is_train=BN_phase,
                                        activation=activation, name='conv9', reuse=reuse)
            conv_9b, m9 = conv2d_layer(conv_9, [conv_size, conv_size, nb_conv, nb_conv],
                                        if_BN=if_BN, is_train=BN_phase,
                                        activation=activation, name='conv9bis', reuse=reuse)
            logits, m9bb = conv2d_layer(conv_9b,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, is_train=BN_phase, name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Unet_encoder_no_BN(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               ):

    with tf.name_scope('Unet5'):
        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv], #[height, width, in_channels, output_channels]
                                     if_BN=False, activation=activation,
                                    name='conv1', reuse=reuse)
            conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                         if_BN=False, activation=activation, name='conv1bis', reuse=reuse)
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                     if_BN=False, activation=activation, name='conv2', reuse=reuse)
            conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                         if_BN=False, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     if_BN=False, activation=activation, name='conv3', reuse=reuse)
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=False, activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=False, activation=activation, name='conv4', reuse=reuse)
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                             activation=activation,
                                             name='conv4bisbis', reuse=reuse)
            conv4_pooling = max_pool_2by2(conv4bis, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16],
                                     if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5', reuse=reuse)
            conv5bis, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16],
                                         if_BN=if_BN, is_train=BN_phase, activation=activation, name='bot5bis', reuse=reuse)
            deconv1, m5u = conv2d_transpose_layer(conv5bis, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv1', reuse=reuse)

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4bis], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv6', reuse=reuse)
            conv_6bis, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                          if_BN=if_BN, is_train=BN_phase,
                                          activation=activation, name='conv6bis', reuse=reuse)
            deconv2, m6u = conv2d_transpose_layer(conv_6bis, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation,
                                                  name='deconv2', reuse=reuse)

            concat2 = concat([deconv2, conv3bis], name='concat2')
            conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv7', reuse=reuse)
            conv_7bis, m7b = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                          if_BN=if_BN, is_train=BN_phase,
                                          activation=activation, name='conv7bis', reuse=reuse)
            deconv3, m7u = conv2d_transpose_layer(conv_7bis, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv3', reuse=reuse)

            concat3 = concat([deconv3, conv2bis], name='concat3')
            conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                      if_BN=if_BN, is_train=BN_phase,
                                      activation=activation, name='conv8', reuse=reuse)
            conv_8bis, m8b = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                          if_BN=if_BN, is_train=BN_phase,
                                          activation=activation,  name='conv8bis', reuse=reuse)
            deconv4, m8u = conv2d_transpose_layer(conv_8bis, shape=[conv_size, conv_size, nb_conv * 2, nb_conv],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv4', reuse=reuse)

            concat4 = concat([deconv4, conv1bis], name='concat4')
            deconv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                        if_BN=if_BN, is_train=BN_phase,
                                        activation=activation, name='conv9', reuse=reuse)
            deconv_9bis, m9b = conv2d_layer(deconv_9, [conv_size, conv_size, nb_conv, nb_conv],
                                            if_BN=if_BN, is_train=BN_phase,
                                            activation=activation, name='conv9bis', reuse=reuse)
            logits, m9bb = conv2d_layer(deconv_9bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, is_train=BN_phase, name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def model_Unet_without_BN(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               device=0,
               ):

    with tf.name_scope('Unet6'):
        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv],  #[height, width, in_channels, output_channels]
                                     if_BN=False, activation=activation,
                                    name='conv1', reuse=reuse)
            conv1bis, m1b = conv2d_layer(conv1, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                         if_BN=False, activation=activation, name='conv1bis', reuse=reuse)
            conv1_pooling = max_pool_2by2(conv1bis, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                     if_BN=False, activation=activation, name='conv2', reuse=reuse)
            conv2bis, m2b = conv2d_layer(conv2, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                         if_BN=False, activation=activation, name='conv2bis', reuse=reuse)
            conv2_pooling = max_pool_2by2(conv2bis, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     if_BN=False, activation=activation, name='conv3', reuse=reuse)
            conv3bis, m3b = conv2d_layer(conv3, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=False, activation=activation, name='conv3bis', reuse=reuse)
            conv3_pooling = max_pool_2by2(conv3bis, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=False, activation=activation, name='conv4', reuse=reuse)
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                             activation=activation,
                                             name='conv4bisbis', reuse=reuse)
            conv4_pooling = max_pool_2by2(conv4bis, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16],
                                     if_BN=False, activation=activation, name='bot5', reuse=reuse)
            conv5bis, m5b = conv2d_layer(conv5, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16],
                                         if_BN=False, activation=activation, name='bot5bis', reuse=reuse)
            deconv1, m5u = conv2d_transpose_layer(conv5bis, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                                  stride=2, if_BN=False,
                                                  activation=activation, name='deconv1', reuse=reuse)

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4bis], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                      if_BN=False,
                                      activation=activation, name='conv6', reuse=reuse)
            conv_6bis, m6b = conv2d_layer(conv_6, [conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                          if_BN=False,
                                          activation=activation, name='conv6bis', reuse=reuse)
            deconv2, m6u = conv2d_transpose_layer(conv_6bis, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                                  stride=2, if_BN=False,
                                                  activation=activation,
                                                  name='deconv2', reuse=reuse)

            concat2 = concat([deconv2, conv3bis], name='concat2')
            conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                      if_BN=False, activation=activation, name='conv7', reuse=reuse)
            conv_7bis, m7b = conv2d_layer(conv_7, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                          if_BN=False, activation=activation, name='conv7bis', reuse=reuse)
            deconv3, m7u = conv2d_transpose_layer(conv_7bis, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv3', reuse=reuse)

            concat3 = concat([deconv3, conv2bis], name='concat3')
            conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                      if_BN=False, activation=activation, name='conv8', reuse=reuse)
            conv_8bis, m8b = conv2d_layer(conv_8, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                          if_BN=False, activation=activation,  name='conv8bis', reuse=reuse)
            deconv4, m8u = conv2d_transpose_layer(conv_8bis, shape=[conv_size, conv_size, nb_conv * 2, nb_conv],
                                                  stride=2, if_BN=False, activation=activation, name='deconv4', reuse=reuse)

            concat4 = concat([deconv4, conv1bis], name='concat4')
            deconv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                        if_BN=False, activation=activation, name='conv9', reuse=reuse)
            deconv_9bis, m9b = conv2d_layer(deconv_9, [conv_size, conv_size, nb_conv, nb_conv],
                                            if_BN=False, activation=activation, name='conv9bis', reuse=reuse)
            logits, m9bb = conv2d_layer(deconv_9bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, name='logits', reuse=reuse)
    print_nodes_name_shape(tf.get_default_graph())
    return logits, []


def model_Unet_with_droupout(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               device=0,
               ):

    with tf.name_scope('Unet7'):
        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv],  #[height, width, in_channels, output_channels]
                                     if_BN=False, activation=activation,
                                     name='conv1', reuse=reuse)
            conv1D = dropout(conv1, drop_prob, 'do1')
            conv1bis, m1b = conv2d_layer(conv1D, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                         if_BN=False, activation=activation, name='conv1bis', reuse=reuse)
            conv1bisD = dropout(conv1bis, drop_prob, 'do1b')
            conv1_pooling = max_pool_2by2(conv1bisD, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                     if_BN=False, activation=activation, name='conv2', reuse=reuse)
            conv2D = dropout(conv2, drop_prob, 'do2')
            conv2bis, m2b = conv2d_layer(conv2D, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                         if_BN=False, activation=activation, name='conv2bis', reuse=reuse)
            conv2bisD = dropout(conv2bis, drop_prob, 'do2b')
            conv2_pooling = max_pool_2by2(conv2bisD, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     if_BN=False, activation=activation, name='conv3', reuse=reuse)
            conv3D = dropout(conv3, drop_prob, 'do3')
            conv3bis, m3b = conv2d_layer(conv3D, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=False, activation=activation, name='conv3bis', reuse=reuse)
            conv3bisD = dropout(conv3bis, drop_prob, 'do3b')
            conv3_pooling = max_pool_2by2(conv3bisD, name='maxp3')

            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=False, activation=activation, name='conv4', reuse=reuse)
            conv4D = dropout(conv4, drop_prob, 'do4')
            conv4bis, m4b = conv2d_layer(conv4D, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                         if_BN=False, activation=activation,
                                         name='conv4bisbis', reuse=reuse)
            conv4bisD = dropout(conv4bis, drop_prob, 'do4b')
            conv4_pooling = max_pool_2by2(conv4bisD, name='maxp4')

        with tf.name_scope('bottom'):
            conv5, m5 = conv2d_layer(conv4_pooling, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 16],
                                     if_BN=if_BN, activation=activation, name='bot5', reuse=reuse)
            conv5D = dropout(conv5, drop_prob, 'do5')
            conv5bis, m5b = conv2d_layer(conv5D, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 16],
                                         if_BN=if_BN, activation=activation, name='bot5bis', reuse=reuse)
            conv5bisD = dropout(conv5bis, drop_prob, 'do5b')

            deconv1, m5u = conv2d_transpose_layer(conv5bisD, shape=[conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                                  stride=2, if_BN=False,
                                                  activation=activation, name='deconv1', reuse=reuse)

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv4bis], name='concat1')
            conv_6, m6 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 16, nb_conv * 8],
                                      if_BN=if_BN,
                                      activation=activation, name='conv6', reuse=reuse)
            conv6D = dropout(conv_6, drop_prob, 'do6')
            conv_6bis, m6b = conv2d_layer(conv6D, [conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                          if_BN=if_BN,
                                          activation=activation, name='conv6bis', reuse=reuse)
            conv6bisD = dropout(conv_6bis, drop_prob, 'do6b')
            deconv2, m6u = conv2d_transpose_layer(conv6bisD, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                                  stride=2, if_BN=False,
                                                  activation=activation,
                                                  name='deconv2', reuse=reuse)

            concat2 = concat([deconv2, conv3bis], name='concat2')
            conv_7, m7 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                      if_BN=if_BN, activation=activation, name='conv7', reuse=reuse)
            conv7D = dropout(conv_7, drop_prob, 'do7')
            conv_7bis, m7b = conv2d_layer(conv7D, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                          if_BN=if_BN, activation=activation, name='conv7bis', reuse=reuse)
            conv7bisD = dropout(conv_7bis, drop_prob, 'do7b')
            deconv3, m7u = conv2d_transpose_layer(conv7bisD, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                  stride=2, if_BN=False, is_train=BN_phase,
                                                  activation=activation, name='deconv3', reuse=reuse)

            concat3 = concat([deconv3, conv2bis], name='concat3')
            conv_8, m8 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                      if_BN=if_BN, activation=activation, name='conv8', reuse=reuse)
            conv8D = dropout(conv_8, drop_prob, 'do8')
            conv_8bis, m8b = conv2d_layer(conv8D, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                          if_BN=if_BN, activation=activation,  name='conv8bis', reuse=reuse)
            conv8bisD = dropout(conv_8bis, drop_prob, 'do8b')
            deconv4, m8u = conv2d_transpose_layer(conv8bisD, shape=[conv_size, conv_size, nb_conv * 2, nb_conv],
                                                  stride=2, if_BN=False, activation=activation, name='deconv4', reuse=reuse)

            concat4 = concat([deconv4, conv1bis], name='concat4')
            conv_9, m9 = conv2d_layer(concat4, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                        if_BN=if_BN, activation=activation, name='conv9', reuse=reuse)
            conv9D = dropout(conv_9, drop_prob, 'do8b')
            conv_9bis, m9b = conv2d_layer(conv9D, [conv_size, conv_size, nb_conv, nb_conv],
                                            if_BN=if_BN, activation=activation, name='conv9bis', reuse=reuse)
            logits, m9bb = conv2d_layer(conv_9bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, name='logits', reuse=reuse)
    print_nodes_name_shape(tf.get_default_graph())
    return logits, []


def model_Unet_with_droupout_shallow(pipeline,
               patch_size,
               batch_size,
               conv_size,
               nb_conv,
               drop_prob,
               if_BN=True,
               BN_phase=None,
               activation='relu',
               reuse=False,
               mode='regression',
               nb_classes=3,
               device=0,
               ):

    with tf.name_scope('Unet8'):
        with tf.name_scope('contractor'):
            conv1, m1 = conv2d_layer(pipeline['img'], shape=[conv_size, conv_size, 1, nb_conv],  #[height, width, in_channels, output_channels]
                                     if_BN=False, activation=activation,
                                     name='conv1', reuse=reuse)
            conv1D = dropout(conv1, drop_prob, 'do1')
            conv1bis, m1b = conv2d_layer(conv1D, shape=[conv_size, conv_size, nb_conv, nb_conv],
                                         if_BN=False, activation=activation, name='conv1bis', reuse=reuse)
            conv1bisD = dropout(conv1bis, drop_prob, 'do1b')
            conv1_pooling = max_pool_2by2(conv1bisD, name='maxp1')

            conv2, m2 = conv2d_layer(conv1_pooling, shape=[conv_size, conv_size, nb_conv, nb_conv * 2],
                                     if_BN=False, activation=activation, name='conv2', reuse=reuse)
            conv2D = dropout(conv2, drop_prob, 'do2')
            conv2bis, m2b = conv2d_layer(conv2D, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                         if_BN=False, activation=activation, name='conv2bis', reuse=reuse)
            conv2bisD = dropout(conv2bis, drop_prob, 'do2b')
            conv2_pooling = max_pool_2by2(conv2bisD, name='maxp2')

            conv3, m3 = conv2d_layer(conv2_pooling, shape=[conv_size, conv_size, nb_conv * 2, nb_conv * 4],
                                     if_BN=False, activation=activation, name='conv3', reuse=reuse)
            conv3D = dropout(conv3, drop_prob, 'do3')
            conv3bis, m3b = conv2d_layer(conv3D, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                         if_BN=False, activation=activation, name='conv3bis', reuse=reuse)
            conv3bisD = dropout(conv3bis, drop_prob, 'do3b')
            conv3_pooling = max_pool_2by2(conv3bisD, name='maxp3')

        with tf.name_scope('bottom'):
            conv4, m4 = conv2d_layer(conv3_pooling, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 8],
                                     if_BN=False, activation=activation, name='bot4', reuse=reuse)
            conv4D = dropout(conv4, drop_prob, 'do4')
            conv4bis, m4b = conv2d_layer(conv4D, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                         if_BN=False, activation=activation, name='bot4bis', reuse=reuse)
            conv4bisD = dropout(conv4bis, drop_prob, 'do4b')

            deconv1, m4u = conv2d_transpose_layer(conv4bisD, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                                  stride=2, if_BN=False,
                                                  activation=activation, name='deconv1', reuse=reuse)

        with tf.name_scope('decontractor'):
            concat1 = concat([deconv1, conv3bis], name='concat1')
            conv_5, m5 = conv2d_layer(concat1, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                      if_BN=False, activation=activation, name='conv5', reuse=reuse)
            conv5D = dropout(conv_5, drop_prob, 'do5')
            conv_5bis, m5b = conv2d_layer(conv5D, [conv_size, conv_size, nb_conv * 4, nb_conv * 4],
                                          if_BN=False, activation=activation, name='conv5bis', reuse=reuse)
            conv5bisD = dropout(conv_5bis, drop_prob, 'do5b')
            deconv2, m5u = conv2d_transpose_layer(conv5bisD, shape=[conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                  stride=2, if_BN=if_BN, is_train=BN_phase,
                                                  activation=activation, name='deconv2', reuse=reuse)

            concat2 = concat([deconv2, conv2bis], name='concat2')
            conv_6, m6 = conv2d_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                      if_BN=False, activation=activation, name='conv6', reuse=reuse)
            conv6D = dropout(conv_6, drop_prob, 'do6')
            conv_6bis, m6b = conv2d_layer(conv6D, [conv_size, conv_size, nb_conv * 2, nb_conv * 2],
                                          if_BN=False, activation=activation,  name='conv6bis', reuse=reuse)
            conv6bisD = dropout(conv_6bis, drop_prob, 'do6b')
            deconv3, m6u = conv2d_transpose_layer(conv6bisD, shape=[conv_size, conv_size, nb_conv * 2, nb_conv],
                                                  stride=2, if_BN=False, activation=activation, name='deconv3', reuse=reuse)

            concat3 = concat([deconv3, conv1bis], name='concat3')
            conv_7, m7 = conv2d_layer(concat3, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                        if_BN=False, activation=activation, name='conv7', reuse=reuse)
            conv7D = dropout(conv_7, drop_prob, 'do6')
            conv_7bis, m7b = conv2d_layer(conv7D, [conv_size, conv_size, nb_conv, nb_conv],
                                            if_BN=False, activation=activation, name='conv7bis', reuse=reuse)
            conv7bisD = dropout(conv_7bis, drop_prob, 'do6b')
            logits, m7bb = conv2d_layer(conv7bisD,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, name='logits', reuse=reuse)
    print_nodes_name_shape(tf.get_default_graph())
    return logits, []


def model_xlearn_like(pipeline,
                 patch_size,
                 batch_size,
                 conv_size,
                 nb_conv,
                 drop_prob,
                 if_BN=True,
                 BN_phase=None,
                 activation='relu',
                 reuse=False,
                 mode='regression',
                 nb_classes=3,
                 ):

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
            conv4bis, m4b = conv2d_layer(conv4, shape=[conv_size, conv_size, nb_conv * 8, nb_conv * 8],
                                             is_train=BN_phase, activation=activation,
                                             name='conv4bis', reuse=reuse)
            conv4bisbis, m4bb = conv2d_layer(conv4bis, shape=[conv_size, conv_size, nb_conv * 8, 1],
                                             is_train=BN_phase, activation=activation,
                                             name='conv4bisbis', reuse=reuse)

        with tf.name_scope('dnn'):
            conv4_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
            full_layer_1, mf1 = normal_full_layer(conv4_flat, patch_size ** 2 // 128, activation=activation,
                                                  if_BN=if_BN, is_train=BN_phase, name='dnn1', reuse=reuse)
            full_dropout1 = dropout(full_layer_1, drop_prob, name='dropout1')
            full_layer_2, mf2 = normal_full_layer(full_dropout1, patch_size ** 2 // 128, activation=activation,
                                                  if_BN=if_BN, is_train=BN_phase, name='dnn2', reuse=reuse)
            full_dropout2 = dropout(full_layer_2, drop_prob, name='dropout2')
            full_layer_3, mf3 = normal_full_layer(full_dropout2, patch_size ** 2 // 64, activation=activation,
                                                  if_BN=if_BN, is_train=BN_phase, name='dnn3', reuse=reuse)
            full_dropout3 = dropout(full_layer_3, drop_prob, name='dropout3')
            dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

        with tf.name_scope('decoder'):
            deconv_5, m5 = conv2d_transpose_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 8],
                                                  [batch_size, patch_size // 8, patch_size // 8, nb_conv * 8],
                                                  if_BN=if_BN, is_train=BN_phase, name='deconv5',
                                                  activation=activation, reuse=reuse)  #[height, width, in_channels, output_channels]
            deconv_5bis, m5b = conv2d_transpose_layer(deconv_5, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size // 8, patch_size // 8, nb_conv * 4],
                                                      if_BN=if_BN, is_train=BN_phase, name='deconv5bis',
                                                      activation=activation, reuse=reuse)
            concat1 = concat([up_2by2(deconv_5bis, name='up1'), conv3bis], name='concat1')

            deconv_6, m6 = conv2d_transpose_layer(concat1, [conv_size, conv_size, nb_conv * 8, nb_conv * 4],
                                                  # fixme: batch_size here might not be automatic while inference
                                                  [batch_size, patch_size // 4, patch_size // 4, nb_conv * 4],
                                                  if_BN=if_BN, is_train=BN_phase, name='deconv6',
                                                  activation=activation, reuse=reuse)
            deconv_6bis, m6b = conv2d_transpose_layer(deconv_6, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size // 4, patch_size // 4, nb_conv * 2],
                                                      if_BN=if_BN, is_train=BN_phase, name='deconv6bis',
                                                      activation=activation, reuse=reuse)
            concat2 = concat([up_2by2(deconv_6bis, name='up2'), conv2bis], name='concat2')

            deconv_7, m7 = conv2d_transpose_layer(concat2, [conv_size, conv_size, nb_conv * 4, nb_conv * 2],
                                                  # fixme: batch_size here might not be automatic while inference
                                                  [batch_size, patch_size // 2, patch_size // 2, nb_conv * 2],
                                                  if_BN=if_BN, is_train=BN_phase, name='deconv7',
                                                  activation=activation, reuse=reuse)
            deconv_7bis, m7b = conv2d_transpose_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size // 2, patch_size //2, nb_conv],
                                                      if_BN=if_BN, is_train=BN_phase, name='deconv7bis',
                                                      activation=activation, reuse=reuse)
            concat3 = concat([up_2by2(deconv_7bis, name='up3'), conv1bis], name='concat3')

            deconv_8, m8 = conv2d_transpose_layer(concat3, [conv_size, conv_size, nb_conv * 2, nb_conv],
                                                  # fixme: batch_size here might not be automatic while inference
                                                  [batch_size, patch_size, patch_size, nb_conv],
                                                  if_BN=if_BN, is_train=BN_phase, name='deconv8',
                                                  activation=activation, reuse=reuse)
            deconv_8bis, m8b = conv2d_transpose_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv],
                                                      # fixme: batch_size here might not be automatic while inference
                                                      [batch_size, patch_size, patch_size, nb_conv],
                                                      if_BN=if_BN, is_train=BN_phase, name='deconv8bis',
                                                      activation=activation, reuse=reuse)

            logits, m8bb = conv2d_transpose_layer(deconv_8bis,
                                        [conv_size, conv_size, nb_conv, 1 if mode == 'regression' else nb_classes],
                                        # fixme: batch_size here might not be automatic while inference
                                        [batch_size, patch_size, patch_size, 1 if mode == 'regression' else nb_classes],
                                        if_BN=False, is_train=BN_phase,
                                        name='logits', reuse=reuse)
        print_nodes_name_shape(tf.get_default_graph())
        return logits, []


def custom(pipeline,
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

    pass


model_dict = {
    'LRCS': model_LRCS,
    'LRCS2': model_LRCS_improved,
    'LRCS3': model_LRCS_constant,
    'LRCS4': model_LRCS_shallow,
    'LRCS5': model_LRCS_simple,
    'LRCS6': model_LRCS_purConv,
    'LRCS7': model_LRCS_LeCun,
    'LRCS8': model_LRCS_Weka,
    'LRCS9': model_LRCS_weka_constant,
    'LRCS10': model_LRCS_lecun_thinner_weka_encoder,
    'LRCS11': model_LRCS_lecun_thinner_encoder,
    'LRCS12': model_LRCS_mix_skipconnect,
    'LRCS13': model_LRCS_dropout_on_conv,
    'LRCS14': model_LRCS_full_FCLs,
    'LRCS15': model_LRCS_deeper_with_dropout_on_conv,
    'Xlearn': model_xlearn_like,
    'Unet': model_Unet,
    'Unet2': model_Unet_shallow,
    'Unet3': model_Unet_weka,
    # 'Unet4': model_Unet_upsample,  # upsampling2d not working
    'Unet5': model_Unet_encoder_no_BN,
    'Unet6': model_Unet_without_BN,
    'Unet7': model_Unet_with_droupout,
    'Unet8': model_Unet_with_droupout_shallow,
    'Segnet': model_Segnet_like,
    'Segnet2': model_Segnet_improved,
    'Segnet3': model_Segnet_constant,
    'Segnet4': model_Segnet_shallow,
    'custom': custom,
}