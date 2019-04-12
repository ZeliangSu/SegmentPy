import h5py
import numpy as np
import tensorflow as tf

def parser(args):
    name, patch_size = args
    print(name)
    name = name.decode('utf-8')
    patch_size = int(patch_size.decode('utf-8'))
    return name


def _pyfn_wrapper(args):
    return tf.py_func(parser,  #wrapped pythonic function
                      [args],
                      [tf.int32]  #[input, output] dtype
                      )

l_a = [i for i in range(90)]
l_b = [10] * 90
a = tf.placeholder(tf.int32, shape=[None])
b = tf.placeholder(tf.int32, shape=[None])
tmp = [(a, b)]
print(tmp)
ds = tf.data.Dataset.from_tensor_slices(tmp)
ds = ds.map(_pyfn_wrapper, num_parallel_calls=5)
ds = ds.batch(5, drop_remainder=True).shuffle(5).prefetch(5).repeat()
it = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)  #one output with shape 1
iter_init_op = it.make_initializer(ds, name='iter')
a_it = it.get_next()
sum = tf.Variable(0)
sum = tf.add(sum, a)

with tf.Session() as sess:
    sess.run([iter_init_op, tf.global_variables_initializer()], feed_dict={a: l_a,  b: l_b})
    # sess.run([iter_init_op])
    for step in range(90):
        print(sess.run(sum))


############################Activation visualization
import tensorflow as tf
import numpy as np

#https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4


