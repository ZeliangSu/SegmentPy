import numpy as np
import h5py
import tensorflow as tf
import threading
import os
import multiprocessing as mp


def MBGDHelper(patch_size, batch_size, is_training=True, ncores=mp.cpu_count()):
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
    from itertools import repeat
    # get length of epoch
    flist = []
    for dirpath, _, fnames in os.walk('./proc/{}/{}/'.format('train' if is_training else 'test', patch_size)):
        for fname in fnames:
            flist.append(os.path.abspath(os.path.join(dirpath, fname)))
    ep_len = len(flist)
    print('Epoch length: {}'.format(ep_len))

    # init list of files
    batch = tf.data.Dataset.from_tensor_slices((tf.constant(flist)))
    batch = batch.map(_pyfn_wrapper, num_parallel_calls=ncores)
    batch = batch.shuffle(batch_size).batch(batch_size, drop_remainder=True).prefetch(ncores).repeat()
    #todo: prefetch_to_device

    # construct iterator
    it = batch.make_initializable_iterator()
    iter_init_op = it.initializer

    # get next img and label
    X_it, y_it = it.get_next()
    inputs = {'img': X_it, 'label': y_it, 'iterator_init_op': iter_init_op}
    return inputs, ep_len

def _folder_parser(directory, is_training, patch_size):
    flist = []
    for dirpath, _, fnames in os.walk('./proc/{}/{}/'.format('train' if is_training else 'test', patch_size)):
        for fname in fnames:
            flist.append(os.path.abspath(os.path.join(dirpath, fname)))
    ep_len = len(flist)
    return flist, ep_len

def parse_h5(name, patch_size):
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
    with h5py.File(name.decode('utf-8'), 'r') as f:
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


def _pyfn_wrapper(filename):
    '''
    input:
    -------
    filename: (tf.data.Dataset)  Tensors of strings

    output:
    -------
    function: (function) tensorflow's pythonic function with its arguements
    '''

    # filename, patch_size = args
    patch_size = 96 #fixme: ask how to tf.data.Dataset map multi-args
    # args = [filename, patch_size]
    return tf.py_func(parse_h5,  #wrapped pythonic function
                      [filename, patch_size],
                      [tf.float32, tf.float32]  #[input, output] dtype
                      )

if __name__ == '__main__':
   pass
