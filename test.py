import tensorflow as tf
import numpy as np
outdir = './proc/'
import h5py
import os


def nn():
    with tf.variable_scope("placeholder"):
        input = tf.placeholder(tf.float32, shape=[None, 10, 10])
        y_true = tf.placeholder(tf.int32, shape=[None, 1])

    with tf.variable_scope('FullyConnected'):
        w = tf.get_variable('w', shape=[10, 10], initializer=tf.random_normal_initializer(stddev=1e-1))
        b = tf.get_variable('b', shape=[10], initializer=tf.constant_initializer(0.1))
        z = tf.matmul(input, w) + b
        y = tf.nn.relu(z)

        w2 = tf.get_variable('w2', shape=[10, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
        b2 = tf.get_variable('b2', shape=[1], initializer=tf.constant_initializer(0.1))
        z = tf.matmul(y, w2) + b2

    with tf.variable_scope('Loss'):
        losses = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(y_true, tf.float32), z)
        loss_op = tf.reduce_mean(losses)

    with tf.variable_scope('Accuracy'):
        y_pred = tf.cast(z > 0, tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
        accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")

    adam = tf.train.AdamOptimizer(1e-2)
    train_op = adam.minimize(loss_op, name="train_op")

    return train_op, loss_op, accuracy

def train(train_op, loss_op, accuracy):
    with tf.Session() as sess:
        # ... init our variables, ...
        sess.run(tf.global_variables_initializer())

        # ... check the accuracy before training (without feed_dict!), ...
        sess.run(accuracy)

        # ... train ...
        for i in range(5000):
            #  ... without sampling from Python and without a feed_dict !
            _, loss = sess.run([train_op, loss_op], feed_dict={})

            # We regularly check the loss
            if i % 500 == 0:
                print('iter:%d - loss:%f' % (i, loss))

        # Finally, we check our final accuracy
        sess.run(accuracy)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parser(tfrecord):
    img_features = tf.parse_single_example(
        tfrecord,
        features={
            'X': tf.FixedLenFeature([], tf.string),
            'y': tf.FixedLenFeature([], tf.string),
        })

    X = tf.decode_raw(img_features['X'], tf.float32)
    y = tf.decode_raw(img_features['y'], tf.float32)
    return X, y

def tfrecordReader(filename):
    dataset = tf.data.TFRecordDataset(filenames=filename, num_parallel_reads=10)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10, 1))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(parser, 10))
    # dataset = dataset.prefetch(buffer_size=2)
    return dataset

class generator_yield:
    def __init__(self, file):
        self.file = file

    def __call__(self):
        with h5py.File(self.file, 'r') as f:
            yield f['X'][:], f['y'][:]

def input_fn(fnames, batch_size):
    batches = fnames.apply(tf.data.experimental.parallel_interleave(lambda filename: tf.data.Dataset.from_generator(
        generator_yield(filename), output_types=tf.float32,
        output_shapes=tf.TensorShape([10, 10])), cycle_length=len_fnames))
    batches.shuffle()
    batches.batch(batch_size)
    return batches

# for i in range(5):
#     with tf.io.TFRecordWriter(outdir + '{}_{}_{}_{}.tfrecord'.format(100, 100, 0, i)) as writer:
#         start = 0
#         end = 10
#         a = np.arange(1000).reshape(10, 10, 10)
#         for j in range(10):
#             # Create a feature
#             feature = {
#                 'X': _bytes_feature(a[j, ].tostring()),
#                 'y': _bytes_feature(a[j, ].tostring())
#             }
#             # Create an example protocol buffer
#             example = tf.train.Example(features=tf.train.Features(feature=feature))
#             # Serialize to string and write on the file
#             writer.write(example.SerializeToString())

fnames = []
for dirpath, _, filenames in os.walk('./proc/'):
    for fname in filenames:
        if fname.endswith('h5'):
            fnames.append(os.path.abspath(os.path.join(dirpath, fname)))
len_fnames = len(fnames)
tf.enable_eager_execution()
fnames = tf.data.Dataset.from_tensor_slices(fnames)





