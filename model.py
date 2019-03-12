import tensorflow as tf
from layers import placeholder, conv2d_layer, max_pool_2by2, reshape, normal_full_layer, dropout, conv2d_transpose_layer,\
up_2by2, concat, optimizer, cal_acc, loss_fn, train_operation

def choose_model(mode='test'):
    pass

def model(patch_size, conv_size, nb_conv):
    # encoder
    hold_prob = tf.placeholder(tf.float32, name='prob_hold')
    X, X_dyn_batsize = placeholder([None, patch_size, patch_size, 1], name='input_placeholder')
    y_true, y_dyn_batsize = placeholder([None, patch_size, patch_size, 1], name='output_placeholder')

    conv1, m1 = conv2d_layer(X, shape=[conv_size, conv_size, 1, nb_conv], name='conv1')  #[height, width, in_channels, output_channels]
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

    # dnn
    conv2_flat = reshape(conv4bisbis, [-1, patch_size ** 2 // 64], name='flatten')
    full_layer_1, mf1 = normal_full_layer(conv2_flat, patch_size ** 2 // 128, name='dnn1')
    full_dropout1 = dropout(full_layer_1, hold_prob, name='dropout1')
    full_layer_2, mf2 = normal_full_layer(full_dropout1, patch_size ** 2 // 128, name='dnn2')
    full_dropout2 = dropout(full_layer_2, hold_prob, name='dropout2')
    full_layer_3, mf3 = normal_full_layer(full_dropout2, patch_size ** 2 // 64, name='dnn3')
    full_dropout3 = dropout(full_layer_3, hold_prob, name='dropout3')
    dnn_reshape = reshape(full_dropout3, [-1, patch_size // 8, patch_size // 8, 1], name='reshape')

    # decoder
    deconv_5, m5 = conv2d_transpose_layer(dnn_reshape, [conv_size, conv_size, 1, nb_conv * 4], X_dyn_batsize, name='deconv5')  #[height, width, in_channels, output_channels]
    deconv_5bis, m5b = conv2d_transpose_layer(deconv_5, [conv_size, conv_size, nb_conv * 4, nb_conv * 8], X_dyn_batsize,  stride=1, name='deconv5bis')  #fixme: strides should be 2
    concat1, outshape = concat([up_2by2(deconv_5bis, name='up1'), conv3], name='concat1')

    deconv_6, m6 = conv2d_transpose_layer(concat1, [conv_size, conv_size, int(outshape[3]), nb_conv * 2], X_dyn_batsize,  name='deconv6')
    deconv_6bis, m6b = conv2d_transpose_layer(deconv_6, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], X_dyn_batsize, stride=1, name='deconv6bis')  #fixme: strides should be 2
    concat2, outshape = concat([up_2by2(deconv_6bis, name='up2'), conv2], name='concat2')

    deconv_7, m7 = conv2d_transpose_layer(concat2, [conv_size, conv_size, int(outshape[3]), nb_conv * 2], X_dyn_batsize, name='deconv7')
    deconv_7bis, m7b = conv2d_transpose_layer(deconv_7, [conv_size, conv_size, nb_conv * 2, nb_conv * 2], X_dyn_batsize, stride=1, name='deconv7bis')  #fixme: strides should be 2
    concat3, outshape = concat([up_2by2(deconv_7bis, name='up3'), conv1], name='concat3')

    deconv_8, m8 = conv2d_transpose_layer(concat3, [conv_size, conv_size, int(outshape[3]), nb_conv], X_dyn_batsize, name='deconv8')
    deconv_8bis, m8b = conv2d_transpose_layer(deconv_8, [conv_size, conv_size, nb_conv, nb_conv], X_dyn_batsize, name='deconv8bis')
    deconv_8bisbis, m8bb = conv2d_transpose_layer(deconv_8bis, [conv_size, conv_size, nb_conv, 1], X_dyn_batsize, name='deconv8bisbis')

    # optimizer/train operation
    #fixme: mysterious constant gradient bug with the following function
    # y_pred = prediction(deconv_8bisbis, 'prediction')
    mse, m_loss = loss_fn(y_true, deconv_8bisbis, name='loss_fn')
    opt = optimizer(0.0001, name='optimizer')
    train_op = train_operation(opt, mse, name='train_op')
    grads = opt.compute_gradients(mse)
    grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
    y_pred = tf.cast(deconv_8bisbis, tf.int32)

    # merged summaries
    m_X = tf.summary.image("input", tf.reshape(X, [-1, patch_size, patch_size, 1]), 1)
    m_y = tf.summary.image("output", tf.reshape(y_true, [-1, patch_size, patch_size, 1]), 1)
    merged = tf.summary.merge([m1, m1b, m2, m2b, m3, m3b, m4, m4b, m4bb, mf1, mf2, mf3,
                               m5, m5b, m6, m6b, m7, m7b, m8, m8b, m8bb, m_loss, m_X, m_y, grad_sum])
    return y_pred, train_op, X, y_true, hold_prob, merged

