import numpy as np
import h5py
import tensorflow as tf
import multiprocessing as mp


def inputpipeline(batch_size, ncores=mp.cpu_count(), suffix=''):
    '''
    tensorflow tf.data input pipeline based helper that return image and label at once

    input:
    -------
    patch_size: (int) pixel length of one small sampling window (patch)
    batch_size: (int) number of images per batch before update parameters

    output:
    -------
    inputs: (dict) output of this func, but inputs of the neural network. A dictionary of img, label and the iterator
    initialization operation
    '''

    # placeholder for list fo files
    with tf.name_scope('input_pipeline' + suffix):
        fnames_ph = tf.placeholder(tf.string, shape=[None], name='fnames_ph')
        patch_size_ph = tf.placeholder(tf.int32, shape=[None], name='patch_size_ph')

        # init list of files
        batch = tf.data.Dataset.from_tensor_slices((fnames_ph, patch_size_ph)) #fixme: nested structure for placeholder
        batch = batch.map(_pyfn_wrapper, num_parallel_calls=ncores)
        batch = batch.shuffle(batch_size).batch(batch_size, drop_remainder=True).prefetch(ncores).repeat()
        #todo: prefetch_to_device
        # batch = batch.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))

        # construct iterator
        it = tf.data.Iterator.from_structure(batch.output_types, batch.output_shapes)
        iter_init_op = it.make_initializer(batch, name='iter_init_op')

        # get next img and label
        X_it, y_it = it.get_next()
        inputs = {'img': X_it,
                  'label': y_it,
                  'iterator_init_op': iter_init_op,
                  'fnames_ph': fnames_ph,
                  'patch_size_ph': patch_size_ph}
        return inputs


def _pyfn_wrapper(fname, patch_size):
    '''
    input:
    -------
    filename: (tf.data.Dataset)  Tensors of strings

    output:
    -------
    function: (function) tensorflow's pythonic function with its arguements
    '''

    return tf.py_func(parse_h5,  #wrapped pythonic function
                      [fname, patch_size],
                      [tf.float32, tf.float32]  #[output, output] dtype
                      )


def parse_h5(fname, patch_size):
    '''
    parser that return the input images and  output labels

    input:
    -------
    name: (bytes literal) file name

    output:
    -------
    X: (numpy ndarray) normalized and reshaped array as dataformat 'NHWC'
    y: (numpy ndarray) normalized and reshaped array as dataformat 'NHWC'
    '''

    with h5py.File(fname.decode('utf-8'), 'r') as f:
        X = f['X'][:].reshape(patch_size, patch_size, 1)
        y = f['y'][:].reshape(patch_size, patch_size, 1)
        return _minmaxscalar(X), y  #can't do minmaxscalar for y


def _minmaxscalar(ndarray, dtype=np.float32):
    '''
    func normalize values of a ndarray into interval of 0 to 1

    input:
    -------
    ndarray: (numpy ndarray) input array to be normalized
    dtype: (dtype of numpy) data type of the output of this function

    output:
    -------
    scaled: (numpy ndarray) output normalized array
    '''
    scaled = np.array((ndarray - np.min(ndarray)) / (np.max(ndarray) - np.min(ndarray)), dtype=dtype)
    return scaled



