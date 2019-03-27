import tensorflow as tf
import numpy as np
import datetime
import tqdm
import csv

from helper import MBGDHelper
from model import model
from layers import cal_acc


# params
patch_size = 40
batch_size = 200  # ps40:>1500 GPU allocation warning ps96:>200 GPU allocation warning
nb_epoch = 20
conv_size = 3
nb_conv = 32
learning_rate = 0.00001
now = datetime.datetime.now()
date = '{}_{}_{}'.format(now.year, now.month, now.day)
gpu_list = ['/gpu:0']

# init model
y_pred, train_op, X, y_true, hold_prob, merged = model(patch_size, conv_size, nb_conv, learning_rate=learning_rate)
cal_acc = cal_acc(y_pred, y_true)

# print number of params
print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

# init helper
mh = MBGDHelper(batch_size=batch_size, patch_size=patch_size)
epoch_len = mh.get_epoch()



# begin session
gpu_options = tf.GPUOptions(visible_device_list='0')
# with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess: # use only CPU
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
                                      allow_soft_placement=True,
                                      log_device_placement=False,
                                      )) as sess:
    # init params
    sess.run(tf.global_variables_initializer())

    # init summary
    writer = tf.summary.FileWriter('./logs/' + date + '/bs{}_ps{}_lr{}_cs{}'.format(batch_size, patch_size, learning_rate, conv_size), sess.graph)

    for ep in range(nb_epoch):
        print('Epoch: {}'.format(ep))

        # begin training
        for step in range(epoch_len // batch_size):
            if step % 1000 == 0:
                print('step:{}'.format(step))
            batch = mh.next_batch()

            with tf.device('/device:XLA_GPU:0'):
                summary, _ = sess.run([merged, train_op], feed_dict={X: batch[0], y_true: batch[1], hold_prob: 0.5})

            with tf.device('/device:XLA_CPU:0'):
                # test accuracy
                accuracy, summ_acc = sess.run(cal_acc, feed_dict={X: batch[0], y_true: batch[1], hold_prob: 1.0})
                tf.summary.merge([merged, summ_acc])
                try:
                    with open('./logs/{}/accuracy_bs{}_ps{}_lr{}_cs{}'.format(date,
                                                                              batch_size,
                                                                              patch_size,
                                                                              learning_rate,
                                                                              conv_size), 'a') as f:
                        csv.writer(f).writerow([step + ep * epoch_len // batch_size, accuracy])
                except:
                    with open('./logs/{}/accuracy_bs{}_ps{}_lr{}_cs{}'.format(date,
                                                                              batch_size,
                                                                              patch_size,
                                                                              learning_rate,
                                                                              conv_size), 'w') as f:
                        csv.writer(f).writerow([step + ep * epoch_len // batch_size, accuracy])

            if step % 1000 == 0:
                print('accuracy:{}'.format(accuracy))
            # merge and write summary
            writer.add_summary(summary, step + ep * epoch_len // batch_size)
        mh.shuffle()

    saver = tf.train.Saver()
    saver.save(sess, './weight/{}_{}.ckpt'.format(patch_size, batch_size))
