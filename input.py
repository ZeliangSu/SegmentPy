import numpy as np
import h5py
import tensorflow as tf
import multiprocessing as mp
import warnings
from PIL import Image
from augmentation import random_aug
from proc import _stride


def inputpipeline(batch_size, ncores=mp.cpu_count(), suffix='', augmentation=False, mode='regression'):
    """
    tensorflow tf.data input pipeline based helper that return image and label at once

    input:
    -------
        batch_size: (int) number of images per batch before update parameters

    output:
    -------
        inputs: (dict) output of this func, but inputs of the neural network. A dictionary of img, label and the iterator
        initialization operation
    """

    warnings.warn('The tf.py_func() will be deprecated at TF2.0, replaced by tf.function() please change later the inputpipeline() in input.py')

    is_training = True if suffix in ['train', 'cv', 'test'] else False

    if is_training:
        # placeholder for list fo files
        with tf.name_scope('input_pipeline_' + suffix):
            fnames_ph = tf.placeholder(tf.string, shape=[None], name='fnames_ph')
            patch_size_ph = tf.placeholder(tf.int32, shape=[None], name='patch_size_ph')

            # init and shuffle list of files
            batch = tf.data.Dataset.from_tensor_slices((fnames_ph, patch_size_ph))
            batch = batch.shuffle(tf.cast(tf.shape(fnames_ph)[0], tf.int64))
            # read data
            if mode == 'regression':
                batch = batch.map(_pyfn_parser_wrapper, num_parallel_calls=ncores)
            elif mode == 'classification':
                batch = batch.map(_pyfn_classification_parser_wrapper, num_parallel_calls=ncores)
            # random augment data
            if augmentation:
                batch = batch.map(_pyfn_aug_wrapper, num_parallel_calls=ncores)
            # shuffle and prefetch batch
            batch = batch.shuffle(batch_size).batch(batch_size, drop_remainder=True).prefetch(ncores).repeat()

            # todo: prefetch_to_device
            # batch = batch.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))

            # construct iterator
            it = tf.data.Iterator.from_structure(batch.output_types, batch.output_shapes)
            iter_init_op = it.make_initializer(batch, name='iter_init_op')
            # get next img and label
            X_it, y_it = it.get_next()

            # dict
            inputs = {'img': X_it,
                      'label': y_it,
                      'iterator_init_op': iter_init_op,
                      'fnames_ph': fnames_ph,
                      'patch_size_ph': patch_size_ph}

    else:
        # inference: 1 img
        with tf.name_scope('input_pipeline_' + suffix):
            # placeholders
            fnames_ph = tf.placeholder(tf.string, shape=[None], name='fnames_ph')
            patch_size_ph = tf.placeholder(tf.int32, shape=[None], name='patch_size_ph')

            # init and shuffle list of files
            batch = tf.data.Dataset.from_tensor_slices((fnames_ph, patch_size_ph))

            # read img and stride
            batch = batch.map(_pyfn_stride_wrapper, num_parallel_calls=ncores)

            # simply batch it
            # note: check the last batch results
            batch = batch.batch(batch_size, drop_remainder=False).prefetch(ncores).repeat()

            # construct iterator
            it = tf.data.Iterator.from_structure(batch.output_types, batch.output_shapes)
            iter_init_op = it.make_initializer(batch, name='iter_init_op')

            # get next img and label
            X_it = it.get_next()

            # dict
            inputs = {'batch': X_it,
                      'iterator_init_op': iter_init_op,
                      'fnames_ph': fnames_ph,
                      'patch_size_ph': patch_size_ph,
                      }

    return inputs


def _pyfn_stride_wrapper(fname, patch_size):
    '''designed for inference'''
    return tf.py_func(
        stride,
        [fname, patch_size],
        [tf.float32]
    )


def stride(fname, patch_size):
    '''designed for inference'''
    # stride outside name_scope
    with Image.open(fname) as img:
        patches = _stride(np.array(img), 1, patch_size)
    return patches


def _pyfn_parser_wrapper(fname, patch_size):
    """
    input:
    -------
        filename: (tf.data.Dataset)  Tensors of strings

    output:
    -------
        function: (function) tensorflow's pythonic function with its arguements
    """
    return tf.py_func(parse_h5,  #wrapped pythonic function
                      [fname, patch_size],
                      [tf.float32, tf.int64]  #[output, output] dtype
                      )


def _pyfn_classification_parser_wrapper(fname, patch_size):
    return tf.py_func(
        parse_h5_one_hot,
        [fname, patch_size],
        [tf.float32, tf.int64]
    )


def _pyfn_aug_wrapper(X_img, y_img):
    """
    input:
    -------
        filename: (tf.data.Dataset)  Tensors of strings

    output:
    -------
        function: (function) tensorflow's pythonic function with its arguements
    """
    return tf.py_func(random_aug,
                      [X_img, y_img],
                      [tf.float32, tf.int64]  #[output, output] dtype
                      )


def parse_h5_one_hot(fname, patch_size):
    with h5py.File(fname.decode('utf-8', 'r'), 'r') as f:
        X = f['X'][:].reshape(patch_size, patch_size, 1)
        y = f['y'][:].reshape(patch_size, patch_size, 1)
        # if y is saved as float, convert to int
        if y.dtype == np.float32:
            y = y.astype(np.int8)
        # get how many classes
        nb_classes = len(np.unique(y))
        _y = np.zeros((*y.shape[:3], nb_classes))
        # one hot
        for i in range(nb_classes):
            _y[np.where(y == i), i] = 50
            # note: {0, 50} might better separate two peaks? but not too difficult to converge at the beginning
        return _minmaxscalar(X), y.astype(np.int64)


def parse_h5(fname, patch_size):
    """
    parser that return the input images and  output labels

    input:
    -------
        name: (bytes literal) file name

    output:
    -------
        X: (numpy ndarray) normalized and reshaped array as dataformat 'NHWC'
        y: (numpy ndarray) normalized and reshaped array as dataformat 'NHWC'
    """

    with h5py.File(fname.decode('utf-8'), 'r') as f:
        X = f['X'][:].reshape(patch_size, patch_size, 1)
        y = f['y'][:].reshape(patch_size, patch_size, 1)
        return _minmaxscalar(X), y.astype(np.int64)  #can't do minmaxscalar for y


def _minmaxscalar(ndarray, dtype=np.float32):
    """
    func normalize values of a ndarray into interval of 0 to 1

    input:
    -------
        ndarray: (numpy ndarray) input array to be normalized
        dtype: (dtype of numpy) data type of the output of this function

    output:
    -------
        scaled: (numpy ndarray) output normalized array
    """
    scaled = np.array((ndarray - np.min(ndarray)) / (np.max(ndarray) - np.min(ndarray)), dtype=dtype)
    return scaled



