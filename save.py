#########save part
import tensorflow as tf
import numpy as np


def wrapper(x, y):
    with tf.name_scope('wrapper'):
        return tf.py_func(dummy, [x, y], [tf.float32, tf.float32])


def dummy(x, y):
    return x, y


X_imgs = np.asarray([np.random.rand(784).reshape(28, 28, 1) for _ in range(100)], dtype=np.float32)
y_imgs = np.asarray([np.random.rand(784).reshape(28, 28, 1) for _ in range(100)], dtype=np.float32)
X_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
y_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])

with tf.name_scope('input'):
    ds = tf.data.Dataset.from_tensor_slices((X_ph, y_ph))
    ds = ds.map(wrapper)
    ds = ds.batch(5)

    it = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
    it_init_op = it.make_initializer(ds, name='it_init_op')
    X_it, y_it = it.get_next()

with tf.name_scope('model'):
    with tf.variable_scope('conv1'):
        W1 = tf.get_variable("W", shape=[3, 3, 1, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        C1 = tf.nn.relu(tf.nn.conv2d(X_it, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)

    with tf.variable_scope('conv2'):
        W2 = tf.get_variable("W", shape=[3, 3, 1, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
        C2 = tf.nn.conv2d(C1, W1, strides=[1, 1, 1, 1], padding='SAME') + b2

with tf.name_scope("operation"):
    loss = tf.reduce_mean(tf.losses.mean_squared_error(
        labels=tf.cast(y_it, tf.int32),
        predictions=C2))
    optimizer = tf.train.AdamOptimizer(0.000001)
    grads = optimizer.compute_gradients(loss)
    train_op = optimizer.apply_gradients(grads)

saver = tf.train.Saver()

with tf.Session() as sess:
    tf.summary.FileWriter('./dummy/test', sess.graph)
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)
    for _ in range(100):
        sess.run([tf.global_variables_initializer(), it_init_op], feed_dict={y_ph: y_imgs, X_ph: X_imgs})
        sess.run([train_op])
        saver.save(sess, './dummy/ckpt/test')