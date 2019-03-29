import numpy as np
import h5py
import tensorflow as tf
import threading
import os
import multiprocessing as mp

class MBGDHelper:
    '''Mini Batch Grandient Descen helper'''
    def __init__(self, batch_size, patch_size):
        self.i = 0
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.epoch_len = self._epoch_len()
        self.order = np.arange(self.epoch_len) #data has been pre-shuffle
        self.onoff = False
    def next_batch(self):
        try:
            try:
                with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
                    tmp = self.order.tolist()[self.i * self.batch_size: (self.i + 1) * self.batch_size]
                    X = f['X'][sorted(tmp)].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
                    y = f['y'][sorted(tmp)].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
                    idx = np.random.permutation(X.shape[0])
                self.i += 1
                return X[idx], y[idx]
            except:
                print('\n***Load last batch')
                with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
                    modulo = f['X'].shape % self.batch_size
                    tmp = self.order.tolist()[-modulo:]
                    X = f['X'][sorted(tmp)].reshape(modulo, self.patch_size, self.patch_size, 1)
                    y = f['y'][sorted(tmp)].reshape(modulo, self.patch_size, self.patch_size, 1)
                    idx = np.random.permutation(X.shape[0])
                self.i += 1
                return X[idx], y[idx]
        except Exception as ex:
            raise ex

    def _epoch_len(self):
        with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
            print('Total epoch number is {}'.format(f['X'].shape[0]))
            return f['X'].shape[0]

    def get_epoch(self):
        return self.epoch_len

    def shuffle(self):
        np.random.shuffle(self.order)
        self.i = 0
        print('shuffled datas')


class MBGD_Helper_v2(object):
    def __init__(self,
                 batch_size,
                 patch_size,
                 coord,
                 max_queue_size=32
                 ):

        # init params
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.flist = self._init_flist()
        self.flist_len = len(self.flist)

        # init fifo queue
        self.max_queue_size = max_queue_size
        self.queue = tf.PaddingFIFOQueue(max_queue_size, ['float32'], shapes=[(None, None)])
        self.queue_size = self.queue.size()
        self.threads = []
        self.coord = coord
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        self.i = 0
        self.onoff = False

    def _init_flist(self):
        flist = []
        for dirpath, _, fnames in os.walk('./proc/'):
            for fname in fnames:
                if fname.startswith('{}_{}'.format(self.patch_size, self.batch_size)):
                    flist.append(fname)
        return flist

    def load_data(self):
        print('thread id: {}'.format(threading.get_ident()))
        with h5py.File('./proc/{}_{}_.h5'.format(self.patch_size, self.batch_size), 'r') as f:
            X = f['X'].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
            y = f['y'].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
            idx = np.random.permutation(X.shape[0])
        yield X[idx], y[idx]

    def dequeue(self, nb_batch=1):
        output = self.queue.dequeue_many(nb_batch)
        return output

    def thread_main(self, sess):
        stop = False
        while not stop:
            iterator = self.load_data()
            for data in iterator:
                while self.queue_size.eval(session=sess) == self.max_queue_size:
                    if self.coord.should_stop():
                        break

                if self.coord.should_stop():
                    stop = True
                    print("Enqueue thread receives stop request.")
                    break
                sess.run(self.enqueue, feed_dict={self.sample_placeholder: data})

    def start_threads(self, sess, n_threads=mp.cpu_count()):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads

class MBGD_Helper_v3:
    def __init__(self, patch_size, batch_size):
        self.patch_size = patch_size
        self.batch_size = batch_size
        self.X_flist, self.y_flist = self._init_flist()
        self.len_flist = len(self.X_flist)

    def fetch(self, X_fname, y_fname):
        record_defaults = [[1], [1]*self.patch_size*self.patch_size*self.batch_size]
        X = tf.read_file(X_fname)
        y = tf.read_file(y_fname)
        X = tf.decode_csv(X, record_defaults=record_defaults, field_delim=',')
        y = tf.decode_csv(y, record_defaults=record_defaults, field_delim=',')
        X = tf.reshape(X, [self.batch_size, self.patch_size, self.patch_size, 1])
        y = tf.reshape(y, [self.batch_size, self.patch_size, self.patch_size, 1])
        return X, y

    def _init_flist(self):
        X_flist = []
        y_flist = []
        for dirpath, _, fnames in os.walk('./proc/'):
            for fname in fnames:
                if fname.startswith('X{}_{}'.format(self.patch_size, self.batch_size)) and \
                        fname.endswith('csv'):
                    X_flist.append(fname)
                elif fname.startswith('y{}_{}'.format(self.patch_size, self.batch_size)) and \
                        fname.endswith('csv'):
                    y_flist.append(fname)
        return X_flist, y_flist

    def load_data(self):
        dataset = tf.data.Dataset.from_tensor_slices((self.X_flist, self.y_flist))
        dataset = dataset.shuffle(self.len_flist)
        dataset = dataset.map(self.fetch, num_parallel_calls=mp.cpu_count())
        dataset = dataset.batch(1)
        dataset = dataset.prefetch(3)
        X, y = dataset.make_one_shot_iterator().get_next()
        return X, y
        # return dataset

class MBGD_Helper_v4:
    def __call__(self, fname, patch_size, batch_size, io):
        with h5py.File(fname, 'r') as f:
            if io == 'X':
                X = f['X'].reshape(batch_size, patch_size, patch_size, 1)
                yield X
            else:
                y = f['y'].reshape(batch_size, patch_size, patch_size, 1)
                yield y

def parse_h5(name, patch_size=40, batch_size=1000):
    '''
    input:
    -------
    name: (bytes literal) file name
    output:
    -------
    X: (numpy ndarray) reshape array as dataformat 'NHWC'
    y: (numpy ndarray) reshape array as dataformat 'NHWC'
    '''
    with h5py.File(name.decode('utf-8'), 'r') as f:
        X = f['X'][:].reshape(batch_size, patch_size, patch_size, 1)
        y = f['y'][:].reshape(batch_size, patch_size, patch_size, 1)
        return X, y


def _pyfn_wrapper(filename):
    return tf.py_func(parse_h5,  #wrapped pythonic function
                      [filename],
                      [tf.float32, tf.int8]  #[input, output] dtype
                      )

def MBGDHelper_v5(patch_size, batch_size, ncores=mp.cpu_count()):
    files = tf.data.Dataset.list_files('./proc/{}_{}_*.h5'.format(patch_size, batch_size))
    dataset = files.map(_pyfn_wrapper, num_parallel_calls=ncores)
    dataset = dataset.batch(1).prefetch(ncores + 1)  #batch() should be 1 here because 1 .h5 file for 1 batch
    it = dataset.make_initializable_iterator()
    iter_init_op = it.initializer
    X_it, y_it = it.get_next()
    inputs = {'img': X_it, 'label': y_it, 'iterator_init_op': iter_init_op}
    return inputs

if __name__ == '__main__':
    mode = 'v3'
    if mode == 'v4':
        from model import model
        patch_size = 40
        batch_size = 1000
        conv_size = 3
        nb_conv = 32
        learning_rate = 0.0001
        fnames = tf.data.Dataset.list_files("./proc/{}_{}_*.h5".format(patch_size, batch_size))
        X_batch = fnames.apply(tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.Dataset.from_generator(
                MBGD_Helper_v4,
                tf.uint8,
                tf.TensorShape([batch_size, patch_size, patch_size, 1]),
                args=(filename, 'X')),
            cycle_length=4,
            block_length=8
            )
        )
        y_batch = fnames.apply(tf.data.experimental.parallel_interleave(
            lambda filename: tf.data.Dataset.from_generator(
                MBGD_Helper_v4,
                tf.uint8,
                tf.TensorShape([1000, 40, 40, 1]),
                args=(filename, 'y')),
            cycle_length=4,
            block_length=8
            )
        )
        y_pred, train_op, X, y_true, hold_prob, merged = model(patch_size, conv_size, nb_conv, learning_rate=learning_rate)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary, _ = sess.run([merged, train_op], feed_dict={X: X_batch, y_true: y_batch, hold_prob: 0.5})

    elif mode == 'v3':
        from model import model
        patch_size = 40
        batch_size = 1000
        conv_size = 3
        nb_conv = 32
        learning_rate = 0.0001
        fnames = tf.data.Dataset.list_files("./proc/{}_{}_*.h5".format(40, 1000))
        # dataset = dataset.shuffle(len(fnames))
        # dataset.map()
