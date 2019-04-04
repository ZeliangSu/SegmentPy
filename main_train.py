import tensorflow as tf
import numpy as np
import datetime
from tqdm import tqdm
import csv
import os

from helper import MBGDHelper
from model import model


# params
patch_size = 96
batch_size = 100  # ps40:>1500 GPU allocation warning ps96:>200 GPU allocation warning
nb_epoch = 20
conv_size = 3
nb_conv = 32
learning_rate = 0.000001  #should use smaller learning rate when decrease batch size
dropout = 0.5
now = datetime.datetime.now()
date = '{}_{}_{}'.format(now.year, now.month, now.day)
hour = '{}'.format(now.hour)
gpu_list = ['/gpu:0']

# init input pipeline
train_inputs, train_len = MBGDHelper(patch_size, batch_size, is_training=True)
test_inputs, test_len = MBGDHelper(patch_size, batch_size, is_training=False)
ep_len = train_len + test_len

# init model
nodes = model(patch_size,
              train_inputs,
              test_inputs,
              batch_size,
              conv_size,
              nb_conv,
              learning_rate=learning_rate,
              drop_prob=dropout
              )

# print number of params
print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))


if not os.path.exists('./logs/{}/'.format(date)):
    os.mkdir('./logs/{}/'.format(date))
    
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
    cv_writer = tf.summary.FileWriter('./logs/{}/hour{}/cv/bs{}_ps{}_lr{}_cs{}'.format(date, hour,
                                                                                       batch_size, patch_size,
                                                                                       learning_rate, conv_size),
                                      sess.graph)
    test_writer = tf.summary.FileWriter('./logs/{}/hour{}/test/bs{}_ps{}_lr{}_cs{}'.format(date, hour,
                                                                                       batch_size, patch_size,
                                                                                       learning_rate, conv_size),
                                      sess.graph)

    for ep in tqdm(range(nb_epoch), desc='Epoch'): #fixme: tqdm print new line after an exception
        sess.run(train_inputs['iterator_init_op'])
        sess.run(test_inputs['iterator_init_op'])
        # begin training
        for step in tqdm(range(train_len // batch_size), desc='Batch step'):
            try:
                # 80%train 10%cross-validation 10%test
                if step % 9 == 8:
                    # 5 percent of the data will be use to cross-validation
                    summary, _ = sess.run([nodes['summary'], nodes['train_or_test_op']], feed_dict={nodes['is_training']: 'cv'})
                    cv_writer.add_summary(summary, step + ep * batch_size)

                    # in situ testing without loading weights like cs-230-stanford
                    summary, _ = sess.run([nodes['summary'], nodes['train_or_test_op']], feed_dict={nodes['is_training']: 'test'})
                    test_writer.add_summary(summary, step + ep * batch_size)

                # 90 percent of the data will be use for training
                else:
                    summary, _ = sess.run([nodes['summary'], nodes['train_or_test_op']],
                                          feed_dict={nodes['is_training']: 'train'})
                    train_writer.add_summary(summary, step + ep * batch_size)

            except tf.errors.OutOfRangeError as e:
                print(e)
                break

        #todo: save model too
        saver = tf.train.Saver()
        saver.save(sess, './weight/{}_{}_{}_epoch{}.ckpt'.format(date, patch_size, batch_size, ep))
