import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
import csv
import os

from helper import MBGDHelper_V6
from model import model
from layers import cal_acc


# params
patch_size = 40
batch_size = 1000  # ps40:>1500 GPU allocation warning ps96:>200 GPU allocation warning
nb_epoch = 20
conv_size = 3
nb_conv = 32
learning_rate = 0.00001
dropout = 0.5
now = datetime.datetime.now()
date = '{}_{}_{}'.format(now.year, now.month, now.day)
hour = '{}'.format(now.hour)
gpu_list = ['/gpu:0']

# init input pipeline
train_inputs, train_len = MBGDHelper_V6(patch_size, batch_size)
test_inputs, test_len = MBGDHelper_V6(patch_size, batch_size)
ep_len = train_len + test_len

# init model
y_pred, train_op, X, y_true, hold_prob, merged, _ = model(patch_size,
                                                                    train_inputs,
                                                                    batch_size,
                                                                    conv_size,
                                                                    nb_conv,
                                                                    learning_rate=learning_rate,
                                                                    drop_prob=dropout,
                                                                    is_training=True
                                                                    )
_, test_op, _, _, _, merged_test, _ = model(patch_size,
                                            test_inputs,
                                            batch_size,
                                            conv_size,
                                            nb_conv,
                                            learning_rate=learning_rate,
                                            drop_prob=1,
                                            is_training=False
                                            )
# cal_acc = cal_acc(y_pred, y_true)

# print number of params
print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

if not os.path.exists('./logs/{}/hour{}/'.format(date, hour)):
    os.mkdir('./logs/{}/hour{}/'.format(date, hour))

# begin session
# with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess: # use only CPU
# gpu_options = tf.GPUOptions(visible_device_list='0')
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
#                                       allow_soft_placement=True,
#                                       log_device_placement=False,
#                                       )) as sess:
with tf.Session() as sess:
    # init params
    sess.run(tf.global_variables_initializer())

    # init summary
    train_writer = tf.summary.FileWriter('./logs/{}/hour{}/train/bs{}_ps{}_lr{}_cs{}'.format(date, hour,
                                                                                             batch_size, patch_size,
                                                                                             learning_rate, conv_size), sess.graph)
    test_writer = tf.summary.FileWriter('./logs/{}/hour{}/test/bs{}_ps{}_lr{}_cs{}'.format(date, hour,
                                                                                           batch_size, patch_size,
                                                                                           learning_rate, conv_size), sess.graph)

    for ep in tqdm(range(nb_epoch), desc='Epoch'): #fixme: tqdm print new line after an exception
        sess.run(train_inputs['iterator_init_op'])
        # begin training
        step_len = ep_len // batch_size
        for step in tqdm(range(ep_len // batch_size), desc='Batch step'): #fixme: handle ep_len not multiple of batch_size

            if step % 10 == 9:
                # 10 percent of the data will be use to testing
                summary = sess.run([merged_test])
                test_writer.add_summary(summary, step + ep * batch_size)

            else:
                # 90 percent of the data will be use for training
                summary, _ = sess.run([merged, train_op])
                train_writer.add_summary(summary, step + ep * batch_size)

        # test accuracy
        # accuracy, summ_acc = sess.run(cal_acc) #fixme: consume again data
        # tf.summary.merge([merged, summ_acc])
        # try:
        #     with open('./logs/{}/hour{}/accuracy_bs{}_ps{}_lr{}_cs{}.csv'.format(date, hour,
        #                                                               batch_size,
        #                                                               patch_size,
        #                                                               learning_rate,
        #                                                               conv_size), 'a') as f:
        #         csv.writer(f).writerow([step + ep * ep_len, accuracy])
        # except:
        #     with open('./logs/{}/hour{}/accuracy_bs{}_ps{}_lr{}_cs{}.csv'.format(date, hour,
        #                                                               batch_size,
        #                                                               patch_size,
        #                                                               learning_rate,
        #                                                               conv_size), 'w') as f:
        #         csv.writer(f).writerow([step + ep * ep_len, accuracy])
        #todo: save model too
        saver = tf.train.Saver()
        saver.save(sess, './weight/{}_{}_{}_epoch{}.ckpt'.format(date, patch_size, batch_size, ep))
