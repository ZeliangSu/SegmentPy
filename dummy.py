import numpy as np
import tensorflow as tf
train = np.arange(909)
test = np.arange(103)

train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(10).batch(10).repeat()
test_ds = tf.data.Dataset.from_tensor_slices(test).shuffle(10).batch(10).repeat()

train_iterator = train_ds.make_initializable_iterator()
test_iterator = test_ds.make_initializable_iterator()

with tf.Session() as sess:
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    for i in range(len(train) + 1):
        print(sess.run(train_iterator.get_next()))
        if i % 9 == 8:
            print(sess.run(test_iterator.get_next()))


