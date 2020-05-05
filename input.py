import numpy as np
import h5py
import tensorflow as tf
import multiprocessing as mp
import warnings
from itertools import product
from PIL import Image
from augmentation import random_aug
from filter import *
import os
import logging
import log

logger = log.setup_custom_logger('root', level=logging.WARNING)
logger.setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)


def inputpipeline_V2(batch_size, ncores=mp.cpu_count(), suffix='', augmentation=False, mode='regression'):
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
            x_coord_ph = tf.placeholder(tf.int32, shape=[None], name='x_coord_ph')
            y_coord_ph = tf.placeholder(tf.int32, shape=[None], name='y_coord_ph')

            # init and shuffle list of files
            batch = tf.data.Dataset.from_tensor_slices((fnames_ph, patch_size_ph, x_coord_ph, y_coord_ph))
            batch = batch.shuffle(tf.cast(tf.shape(fnames_ph)[0], tf.int64))

            # read data
            if mode == 'regression':
                batch = batch.map(_pyfn_regression_parser_wrapper, num_parallel_calls=ncores)
            elif mode == 'classification':
                batch = batch.map(_pyfn_classification_parser_wrapper_V2, num_parallel_calls=ncores)
            elif mode == 'weka':
                batch = batch.map(_pyfn_classification_parser_wrapper_weka, num_parallel_calls=ncores)

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
                      'patch_size_ph': patch_size_ph,
                      'x_coord_ph': x_coord_ph,
                      'y_coord_ph': y_coord_ph}

    else:
        raise NotImplementedError('Inference input need to be debugged')

    return inputs


def _pyfn_regression_parser_wrapper(fname, patch_size):
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
                      [tf.float32, tf.int32]  #[output, output] dtype
                      )


def _pyfn_classification_parser_wrapper_V2(fname, patch_size, x_coord, y_coord):
    """
    input:
    -------
        filename: (tf.data.Dataset)  Tensors of strings

    output:
    -------
        function: (function) tensorflow's pythonic function with its arguements
    """
    return tf.py_func(parse_h5_one_hot_V2,  #wrapped pythonic function
                      [fname, patch_size, x_coord, y_coord],
                      [tf.float32, tf.int32]  #[output, output] dtype
                      )


def _pyfn_classification_parser_wrapper_weka(fname, patch_size, x_coord, y_coord):
    """
    input:
    -------
        filename: (tf.data.Dataset)  Tensors of strings

    output:
    -------
        function: (function) tensorflow's pythonic function with its arguements
    """
    return tf.py_func(parse_h5_one_hot_V3,  #wrapped pythonic function
                      [fname, patch_size, x_coord, y_coord],
                      [tf.float32, tf.int32]  #[output, output] dtype
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
                      [tf.float32, tf.int32]  #[output, output] dtype
                      )


def parse_h5_one_hot_V2(fname, window_size, x_coord, y_coord):
    img = np.asarray(Image.open(fname))
    label = np.asarray(Image.open(fname.decode('utf8').replace('.tif', '_label.tif')))
    assert img.shape == label.shape, 'img and label shape should be equal'
    assert img.shape[0] >= x_coord + window_size, 'window is out of zone'
    assert img.shape[1] >= y_coord + window_size, 'window is out of zone'
    X = np.expand_dims(img[x_coord: x_coord + window_size, y_coord: y_coord + window_size], axis=2)
    y = np.expand_dims(label[x_coord: x_coord + window_size, y_coord: y_coord + window_size], axis=2)
    y = _one_hot(y)
    # logger.debug('y shape: {}, nb_class: {}'.format(y.shape, y.shape[-1]))  # B, H, W, C
    return X, y.astype(np.int32)


def parse_h5_one_hot_V3(fname, window_size, x_coord, y_coord):
    img = np.asarray(Image.open(fname))
    label = np.asarray(Image.open(fname.decode('utf8').replace('.tif', '_label.tif')))
    assert img.shape == label.shape, 'img and label shape should be equal'
    assert img.shape[0] >= x_coord + window_size, 'window is out of zone'
    assert img.shape[1] >= y_coord + window_size, 'window is out of zone'

    # note: the order of the following list shouldn't be changed either training or testing
    l_func = [
        Gaussian_Blur,
        Sobel,
        Hessian,
        DoG,
        Gabor,
        # 'membrane_proj': Membrane_proj,
        Anisotropic_Diffusion1, #no effect
        Anisotropic_Diffusion2,  #no effect
        Bilateral,  #no effect
        Median,
    ]

    # compute feature maps
    _X = img[x_coord: x_coord + window_size, y_coord: y_coord + window_size]
    X = [_X]
    for func in l_func:
        X.append(func(_X))
    X = np.stack(X, axis=2).astype(np.float32)

    # y
    y = np.expand_dims(label[x_coord: x_coord + window_size, y_coord: y_coord + window_size], axis=2)
    y = _one_hot(y)
    # logger.debug('y shape: {}, nb_class: {}'.format(y.shape, y.shape[-1]))  # B, H, W, C
    return X, y.astype(np.int32)


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
        return _minmaxscalar(X), y.astype(np.int32)  #can't do minmaxscalar for y
        # return X, y.astype(np.int32)  #can't do minmaxscalar for y


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


def _one_hot(tensor):
    ''' (batch, H, W) --> one hot to --> (batch, H, W, nb_class)'''
    assert isinstance(tensor, np.ndarray), 'Expect input as a np ndarray'
    # logger.debug('input tensor shape:{}, unique: {}'.format(tensor.shape, np.unique(tensor)))
    if tensor.ndim == 4:
        #note: (Batch, H, W, C)
        tensor = tensor.astype(np.int32)
        # get how many classes
        nb_classes = len(np.unique(tensor))
        # one hot
        out = []
        for i in range(nb_classes):
            tmp = np.zeros(tensor.shape)
            out.append(tmp)
        # stack along the last channel
        out = np.concatenate(out, axis=3)

    elif tensor.ndim == 3:
        #note: (H, W, C) no batch size
        tensor = tensor.astype(np.int32)
        # get how many classes
        nb_classes = len(np.unique(tensor))
        # one hot
        out = []
        for i in range(nb_classes):
            tmp = np.zeros(tensor.shape)
            tmp[np.where(tensor == i)] = 1
            # tmp[np.where(tensor == i)] = 5  # uncomment this line to do 5-hot
            out.append(tmp)
        # stack along the last channel
        out = np.concatenate(out, axis=2)
    else:
        logger.warning('Oupss!')
        raise NotImplementedError('Oupss!')

    # logger.debug('np.shape(out): {}, unique: {}'.format(np.shape(out), np.unique(out)))
    return out


def _inverse_one_hot(tensor):
    assert tensor.ndim == 4, 'Expected a tensor of shape (batch, H, W, class)'
    if tensor.dtype == np.float or tensor.dtype == np.float32 or tensor.dtype == np.float64:
        # make a vote
        output = np.argmax(tensor, axis=3)
        output = np.expand_dims(output, axis=3)

    elif tensor.dtype == np.int or tensor.dtype == np.int8 or tensor.dtype == np.int32 or tensor.dtype == np.int64:
        output = np.argmax(tensor, axis=3)
        output = np.expand_dims(output, axis=3)

    else:
        raise NotImplementedError('Implement a mechanism that if several classes is possible, randomly choose one')

    return output.astype(np.int32)


class coords_gen:
    def __init__(self, train_dir=None, test_dir=None, window_size=512, train_test_ratio=0.9, stride=1, batch_size=None, nb_batch=None):
        self.stride = stride
        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size
        self.window_size = window_size
        self.totest_img = None

        # train id without specified testset repo indicated
        if isinstance(train_dir, str):
            if not train_dir.endswith('/'):
                self.list_train_fname = [train_dir]
            else:
                self.list_train_fname = os.listdir(train_dir)
                self.list_train_fname = [train_dir + relative for relative in self.list_train_fname if not relative.endswith('_label.tif')]
        elif isinstance(train_dir, list):
            self.list_train_fname = train_dir
        else:
            raise TypeError('fname should be a string of path or list of .tif file path strings')

        # train and test ids with testset repo path indicated
        if test_dir is not None:
            # generate indices for testset data if another repo is indicated
            if isinstance(test_dir, str):
                if not test_dir.endswith('/'):
                    self.list_test_fname = [test_dir]
                else:
                    self.list_test_fname = os.listdir(test_dir)
                    self.list_test_fname = [test_dir + relative for relative in self.list_test_fname if not relative.endswith('_label.tif')]
            elif isinstance(test_dir, list):
                self.list_test_fname = test_dir
            else:
                raise TypeError('fname should be a string of path or list of .tif file path strings')
            self.list_test_shapes = self.get_shapes(self.list_test_fname)
            self.test_id = self.id_gen(self.list_test_shapes, self.window_size, self.stride)
            self.totest_img, self.test_list_ps, self.test_xcoord, self.test_ycoord = self.generate_lists(
                id=self.test_id, list_fname=self.list_test_fname, seed=42
            )

        self.train_list_shapes = self.get_shapes(self.list_train_fname)
        self.train_id = self.id_gen(self.train_list_shapes, self.window_size, self.stride)
        self.totrain_img, self.train_list_ps, self.train_xcoord, self.train_ycoord = self.generate_lists(
            id=self.train_id, list_fname=self.list_train_fname, seed=42
        )
        self.nb_batch = nb_batch

    def get_shapes(self, list_fname):
        list_shapes = []
        for fname in list_fname:
            list_shapes.append(np.asarray(Image.open(fname)).shape)
        return list_shapes

    def id_gen(self, list_shapes, window_size, stride):
        # [(0, 1, 2), (0, 1, 3)...]
        id_list = []
        for i, shape in enumerate(list_shapes):
            nb_x = (shape[0] - window_size) // stride + 1
            nb_y = (shape[1] - window_size) // stride + 1
            for x_coord, y_coord in product(range(nb_x), range(nb_y)):
                id_list.append((i, x_coord, y_coord))
        return id_list

    def get_nb_batch(self):
        if self.nb_batch is None:
            return int(len(self.train_id) * self.train_test_ratio // self.batch_size)
        else:
            return self.nb_batch

    def generate_lists(self, id, list_fname, seed=42):
        # fname
        # patch
        # xcoord
        # ycoord
        _imgs = []
        _list_ps = []
        _list_xcoord = []
        _list_ycoord = []

        np.random.seed(seed)
        for n in id:
            i, x, y = n
            _imgs.append(list_fname[i])
            _list_ps.append(self.window_size)
            _list_xcoord.append(x)
            _list_ycoord.append(y)

        # list --> array (--> shuffle) --> list
        _imgs = np.asarray(_imgs)
        _list_ps = np.asarray(_list_ps).astype('int32')
        _list_xcoord = np.asarray(_list_xcoord).astype('int32')
        _list_ycoord = np.asarray(_list_ycoord).astype('int32')
        idx = np.random.permutation(len(_imgs))

        _imgs = _imgs[idx]
        _list_ps = _list_ps[idx]
        _list_xcoord = _list_xcoord[idx]
        _list_ycoord = _list_ycoord[idx]
        return _imgs, _list_ps, _list_xcoord, _list_ycoord

    def shuffle(self):
        tmp = self.get_nb_batch()
        idx = np.random.permutation(tmp)
        self.totrain_img[:tmp] = self.totrain_img[idx]
        self.train_list_ps[:tmp] = self.train_list_ps[idx]
        self.train_xcoord[:tmp] = self.train_xcoord[idx]
        self.train_ycoord[:tmp] = self.train_ycoord[idx]

        if self.totest_img is not None:
            idx = np.random.permutation(len(self.totest_img))
            self.totest_img = self.totest_img[idx]
            self.totest_img = self.totest_img[idx]
            self.totest_img = self.totest_img[idx]
            self.totest_img = self.totest_img[idx]

    def get_train_args(self):
        if self.nb_batch is not None:
            tmp = int(self.nb_batch)
            return self.totrain_img[: tmp], \
                   self.train_list_ps[: tmp], \
                   self.train_xcoord[: tmp], \
                   self.train_ycoord[: tmp]
        else:
            tmp = (int(self.get_nb_batch()))
            return self.totrain_img[: tmp], \
                   self.train_list_ps[: tmp], \
                   self.train_xcoord[: tmp], \
                   self.train_ycoord[: tmp]

    def get_test_args(self):
        if self.totest_img is None:
            tmp = int(self.nb_batch) if self.nb_batch is not None else (int(self.get_nb_batch()))

            return self.totrain_img[tmp:], \
                   self.train_list_ps[tmp:], \
                   self.train_xcoord[tmp:], \
                   self.train_ycoord[tmp:]

        else:
            tmp = int(self.nb_batch) if self.nb_batch is not None else (int(self.get_nb_batch()))

            return self.totest_img[tmp:], \
                   self.test_list_ps[tmp:], \
                   self.test_xcoord[tmp:], \
                   self.test_ycoord[tmp:]

