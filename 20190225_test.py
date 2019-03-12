import tensorflow as tf
import numpy as np
import h5py
import datetime

from helper import MBGDHelper
from model import model
from layers import cal_acc


# params
patch_size = 40
batch_size = 800  # ps40:>1500 GPU allocation warning ps96:>200 GPU allocation warning
nb_epoch = 20
conv_size = 3
nb_conv = 32
now = datetime.datetime.now()
date = '{}_{}_{}'.format(now.year, now.month, now.day)

# init model
y_pred, train_op, X, y_true, hold_prob, merged = model(patch_size, conv_size, nb_conv)
cal_acc = cal_acc(y_pred, y_true)

# print number of params
print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

# init helper
mh = MBGDHelper(batch_size=batch_size, patch_size=patch_size)
epoch = mh.get_epoch()



# begin session
# with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess: # use only CPU
with tf.Session() as sess:
    for epoch in range(nb_epoch):
        print('Epoch: {}'.format(epoch))
        # init params
        sess.run(tf.global_variables_initializer())

        # init summary
        writer = tf.summary.FileWriter('./logs/' + date + '/', sess.graph)

        # begin training
        for i in range(epoch // batch_size + 1):
            if i % 1000 == 0:
                print('step:{}'.format(i))
            batch = mh.next_batch()
            summary, _ = sess.run([merged, train_op], feed_dict={X: batch[0], y_true: batch[1], hold_prob: 0.5})

            # test accuracy
            accuracy, summ_acc = sess.run(cal_acc, feed_dict={X: batch[0], y_true: batch[1], hold_prob: 1.0})
            tf.summary.merge([merged, summ_acc])

            if i % 30 == 0:
                print('accuracy:{}'.format(accuracy))
            # merge and write summary
            writer.add_summary(summary, i)
        mh.shuffle()

    saver = tf.train.Saver()
    saver.save(sess, './weight/{}_{}.ckpt'.format(patch_size, batch_size))
