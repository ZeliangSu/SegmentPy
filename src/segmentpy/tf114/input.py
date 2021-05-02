import numpy as np
import tensorflow as tf
from scipy.ndimage.interpolation import map_coordinates
from scipy.interpolate import interp2d
from itertools import product
from PIL import Image
from segmentpy.tf114.augmentation import random_aug
from segmentpy.tf114.filter import *
from segmentpy.tf114.util import load_img, check_raw_gt_pair
import os

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)


def inputpipeline_V2(batch_size, ncores=mp.cpu_count(),
                     suffix='', augmentation=False, mode='regression',
                     ):
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

    logger.warn('The tf.py_func() will be deprecated at TF2.0, replaced by tf.function() please change later the inputpipeline() in input.py')

    is_training = True if suffix in ['train', 'cv', 'test'] else False

    if is_training:
        # placeholder for list fo files
        with tf.name_scope('input_pipeline_' + suffix):
            fnames_ph = tf.placeholder(tf.string, shape=[None], name='fnames_ph')
            patch_size_ph = tf.placeholder(tf.int32, shape=[None], name='patch_size_ph')
            x_coord_ph = tf.placeholder(tf.int32, shape=[None], name='x_coord_ph')
            y_coord_ph = tf.placeholder(tf.int32, shape=[None], name='y_coord_ph')
            correction_ph = tf.placeholder(tf.float32, shape=[None], name='correction_ph')
            max_nb_cls_ph = tf.placeholder(tf.int32, shape=[None], name='max_nb_cls_ph')
            stretch_ph = tf.placeholder(tf.float32, shape=[None], name='stretch_ph')

            # init and shuffle list of files
            batch = tf.data.Dataset.from_tensor_slices((fnames_ph, patch_size_ph, x_coord_ph, y_coord_ph,
                                                        correction_ph, max_nb_cls_ph, stretch_ph))
            # batch = batch.shuffle(buffer_size=tf.shape(fnames_ph)[0])
            batch = batch.shuffle(buffer_size=tf.cast(tf.shape(fnames_ph)[0], tf.int64))
            # tf.print(tf.cast(tf.shape(fnames_ph)[0], tf.int64))
            # batch = batch.shuffle(buffer_size=batch_size)
            # note: above line is no more necessary if we shuffle pythonically the fnames at the beginning of epoch
            # note: the above line raise 'buffer_size must be greater than zero.' due to the number of image greater than max of int64

            # read data
            if mode == 'regression':
                raise NotImplementedError('The regression is no more supported')
            elif mode == 'classification':
                batch = batch.map(_pyfn_classification_parser_wrapper_V2, num_parallel_calls=ncores)
            elif mode == 'feature_extractors':
                batch = batch.map(_pyfn_classification_parser_wrapper_weka, num_parallel_calls=ncores)
                raise NotImplementedError

            # random augment data
            if augmentation:
                batch = batch.map(_pyfn_aug_wrapper, num_parallel_calls=ncores)

            # shuffle and prefetch batch
            batch = batch.shuffle(batch_size).batch(batch_size, drop_remainder=True).prefetch(ncores).repeat()

            # todo: prefetch_to_device
            # batch = batch.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))

            # construct iterator
            it = tf.data.Iterator.from_structure(
                batch.output_types,
                batch.output_shapes
                # (tf.TensorShape([None, None, None, 1]), tf.TensorShape([None, None, None, 3]))
            )

            iter_init_op = it.make_initializer(batch, name='iter_init_op')
            # get next img and label
            X_it, y_it = it.get_next()

            # X_it = tf.reshape(X_it, [batch_size, patch_size, patch_size, 1])
            # y_it = tf.reshape(X_it, [batch_size, patch_size, patch_size, 1])
            # X_it = tf.reshape(X_it, [batch_size, patch_size, patch_size, -1])
            # y_it = tf.reshape(X_it, [batch_size, patch_size, patch_size, -1])

            # dict
            inputs = {'img': X_it,
                      'label': y_it,
                      'iterator_init_op': iter_init_op,
                      'fnames_ph': fnames_ph,
                      'patch_size_ph': patch_size_ph,
                      'x_coord_ph': x_coord_ph,
                      'y_coord_ph': y_coord_ph,
                      'correction_ph': correction_ph,
                      'max_nb_cls_ph': max_nb_cls_ph,
                      'stretch_ph': stretch_ph,
                      }

    else:
        raise NotImplementedError('Inference input need to be debugged')

    return inputs


def _pyfn_classification_parser_wrapper_V2(fname, patch_size, x_coord, y_coord,
                                           correction, max_nb_cls, stretch):
    """
    input:
    -------
        filename: (tf.data.Dataset)  Tensors of strings

    output:
    -------
        function: (function) tensorflow's pythonic function with its arguements
    """
    return tf.py_func(parse_h5_one_hot_V2,  #wrapped pythonic function
                      [fname, patch_size, x_coord, y_coord, correction, max_nb_cls, stretch],  #fixme: max number of class should be automatic
                      [tf.float32, tf.int32]  #[output, output] dtype
                      )


def _pyfn_classification_parser_wrapper_weka(fname, patch_size, x_coord, y_coord,
                                             ):
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


def parse_h5_one_hot_V2(fname, window_size, x_coord, y_coord, correction=1e3, impose_nb_cls=3, stretch=2.0):
    img = np.asarray(Image.open(fname))
    label = np.asarray(Image.open(fname.decode('utf8').replace('.tif', '_label.tif')))
    logger.debug('fn, ws, x, y: {}, {}, {}, {}'.format(fname, window_size, x_coord, y_coord))
    logger.debug('crt, cls, stch: {}, {}, {}'.format(correction, impose_nb_cls, stretch))
    assert img.shape == label.shape, 'img and label shape should be equal'
    assert img.shape[0] >= x_coord + window_size, 'window is out of zone'
    assert img.shape[1] >= y_coord + window_size, 'window is out of zone'

    if not stretch:
        X = np.expand_dims(img[x_coord: x_coord + window_size, y_coord: y_coord + window_size], axis=2)
        y = np.expand_dims(label[x_coord: x_coord + window_size, y_coord: y_coord + window_size], axis=2)
        y = _one_hot(y, impose_nb_cls=impose_nb_cls)

    else:
        X, y = stretching(img, label=label,
                          x_coord=x_coord, y_coord=y_coord,
                          window_size=window_size, stretch_max=stretch)
        X = np.expand_dims(X, axis=2)
        y = np.expand_dims(y, axis=2)
        y = _one_hot(y, impose_nb_cls=impose_nb_cls)

    logger.debug('y shape: {}, nb_class: {}'.format(y.shape, y.shape[-1]))  # H, W, C
    # return X, y.astype(np.int32)
    # return _minmaxscalar(X), y.astype(np.int32)

    # note: multiplication train more flexible network than minmaxscalar since their might be variation of grayscale
    return X * correction, y.astype(np.int32)


def parse_h5_one_hot_V3(fname, window_size, x_coord, y_coord):
    img = np.asarray(Image.open(fname))
    label = np.asarray(Image.open(fname.decode('utf8').replace('.tif', '_label.tif')))
    assert img.shape == label.shape, 'img and label shape should be equal'
    assert img.shape[0] >= x_coord + window_size, 'window is out of zone'
    assert img.shape[1] >= y_coord + window_size, 'window is out of zone'
    logger.debug('fn, ws, x, y: {}, {}, {}, {}'.format(fname, window_size, x_coord, y_coord))

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


def _one_hot(tensor, impose_nb_cls=None):
    ''' (batch, H, W) --> one hot to --> (batch, H, W, nb_class)'''
    assert isinstance(tensor, np.ndarray), 'Expect input as a np ndarray'
    # logger.debug('input tensor shape:{}, unique: {}'.format(tensor.shape, np.unique(tensor)))

    # get how many classes
    tensor = tensor.astype(np.int32)
    if impose_nb_cls is not None:
        nb_classes = impose_nb_cls
    else:
        nb_classes = len(np.unique(tensor))
    logger.debug('impose: {}, nb_cls: {}'.format(impose_nb_cls, nb_classes))

    if tensor.ndim == 4:
        #note: (Batch, H, W, 1)
        # one hot
        out = []
        for i in range(nb_classes):
            tmp = np.zeros_like(tensor)
            tmp[np.where(tensor == i)] = 1
            out.append(tmp)
        # stack along the last channel
        out = np.concatenate(out, axis=3)

    elif tensor.ndim == 3:
        #note: (H, W, C) no batch size
        # one hot
        out = []
        for i in range(nb_classes):
            tmp = np.zeros(tensor.shape)
            tmp[np.where(tensor == i)] = 1
            out.append(tmp)
        # stack along the last channel
        out = np.concatenate(out, axis=2)
    else:
        logger.warning('Oupss!')
        raise NotImplementedError('Oupss!')

    logger.debug('np.shape(out): {}, unique: {}'.format(np.shape(out), np.unique(out)))
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
    def __init__(self, train_dir=None, valid_dir=None, window_size=512, train_test_ratio=0.9, stride=5, batch_size=None, nb_batch=None):
        self.stride = stride
        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size
        self.window_size = window_size
        self.tovalid_img = None

        # train id without specified validtestset repo indicated
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

        # train and valid ids with validset repo path indicated
        if valid_dir is not None:
            # generate indices for validset data if another repo is indicated
            if isinstance(valid_dir, str):
                if not valid_dir.endswith('/'):
                    self.list_valid_fname = [valid_dir]
                else:
                    self.list_valid_fname = os.listdir(valid_dir)
                    self.list_valid_fname = [valid_dir + relative for relative in self.list_valid_fname if not relative.endswith('_label.tif')]
            elif isinstance(valid_dir, list):
                self.list_valid_fname = valid_dir
            else:
                raise TypeError('fname should be a string of path or list of .tif file path strings')
            self.valid_list_shapes = self.get_shapes(self.list_valid_fname)
            self.valid_id = self.id_gen(self.valid_list_shapes, self.window_size, self.stride)
            self.tovalid_img, self.valid_list_ps, self.valid_xcoord, self.valid_ycoord = self.generate_lists(
                id=self.valid_id, list_fname=self.list_valid_fname, seed=42
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
            self.nb_batch = int(len(self.train_id) * self.train_test_ratio // self.batch_size)
            logger.debug('nb_batch: {}'.format(self.nb_batch))
            return self.nb_batch
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
        logger.debug('fns: {}, ps: {}, xc: {}, yc: {}'.format(_imgs, _list_ps, _list_xcoord, _list_ycoord))
        logger.info('fns len: {}, ps len: {}, xc len: {}, yc len: {}'.format(len(_imgs), len(_list_ps), len(_list_xcoord), len(_list_ycoord)))
        return _imgs, _list_ps, _list_xcoord, _list_ycoord

    def shuffle(self):
        tmp = self.get_nb_batch()
        idx = np.random.permutation(tmp)
        self.totrain_img[:tmp] = self.totrain_img[idx]
        self.train_list_ps[:tmp] = self.train_list_ps[idx]
        self.train_xcoord[:tmp] = self.train_xcoord[idx]
        self.train_ycoord[:tmp] = self.train_ycoord[idx]

        if self.tovalid_img is not None:
            idx = np.random.permutation(len(self.tovalid_img))
            self.tovalid_img = self.tovalid_img[idx]
            self.tovalid_img = self.tovalid_img[idx]
            self.tovalid_img = self.tovalid_img[idx]
            self.tovalid_img = self.tovalid_img[idx]

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

    def get_valid_args(self):
        if self.tovalid_img is None:
            tmp = int(self.nb_batch) if self.nb_batch is not None else (int(self.get_nb_batch()))

            return self.totrain_img[tmp:], \
                   self.train_list_ps[tmp:], \
                   self.train_xcoord[tmp:], \
                   self.train_ycoord[tmp:]

        else:
            # tmp = int(self.nb_batch) if self.nb_batch is not None else (int(self.get_nb_batch()))

            return self.tovalid_img, \
                   self.valid_list_ps, \
                   self.valid_xcoord, \
                   self.valid_ycoord

    def get_min_dim(self):
        train_min_dim = None
        for shape in self.train_list_shapes:
            tmp = min(*shape)
            if train_min_dim is None:
                train_min_dim = tmp
            else:
                if np.less(tmp, train_min_dim):
                    train_min_dim = tmp

        if hasattr(self, 'list_valid_fname'):
            valid_min_dim = None
            for shape in self.valid_list_shapes:
                tmp = min(*shape)
                if valid_min_dim is None:
                    valid_min_dim = tmp
                else:
                    if np.less(tmp, valid_min_dim):
                        valid_min_dim = tmp
            return train_min_dim, valid_min_dim
        return train_min_dim


def stretching(img: np.ndarray,
               x_coord: int,
               y_coord: int,
               window_size: int,
               stretch_max: float,
               label=None,
               ):
    stretch_param = np.random.random() * (stretch_max - 0.5) + 0.5  # e.g. from 0.5 - 2
    a, b = img.shape[0], img.shape[1]

    # find the stretching coordinations
    # left-top corner
    x0 = (
            x_coord
            + ((2 * np.random.random() - 1) * stretch_param)
            * window_size
    )
    if np.less(x0, 0):
        x0 = 0
    elif np.greater(x0, a):
        x0 = a

    # left-bottom corner
    x2 = (
            x_coord
            + ((2 * np.random.random() - 1) * stretch_param)
            * window_size
    )
    if np.less(x2, 0):
        x2 = 0
    elif np.greater(x2, a):
        x2 = a

    # right-top corner
    x1 = (
            x_coord
            + window_size
            + ((2 * np.random.random() - 1) * stretch_param)
            * window_size
    )
    if np.less(x1, 0):
        x1 = 0
    elif np.greater(x1, a):
        x1 = a

    # right-bottom corner
    x3 = (
            x_coord
            + window_size
            + ((2 * np.random.random() - 1) * stretch_param)
            * window_size
    )
    if np.less(x3, 0):
        x3 = 0
    elif np.greater(x3, a):
        x3 = a

    # left-top corner
    y0 = (
            y_coord
            + ((2 * np.random.random() - 1) * stretch_param)
            * window_size
    )
    if np.less(y0, 0):
        y0 = 0
    elif np.greater(y0, b):
        y0 = b

    # left-bottom corner
    y1 = (
            y_coord
            + ((2 * np.random.random() - 1) * stretch_param)
            * window_size
    )
    if np.less(y1, 0):
        y1 = 0
    elif np.greater(y1, b):
        y1 = b

    # right-top corner
    y2 = (
            y_coord
            + window_size
            + ((2 * np.random.random() - 1) * stretch_param)
            * window_size
    )
    if np.less(y2, 0):
        y2 = 0
    elif np.greater(y2, b):
        y2 = b

    # right-bottom corner
    y3 = (
            y_coord
            + window_size
            + ((2 * np.random.random() - 1) * stretch_param)
            * window_size
    )
    if np.less(y3, 0):
        y3 = 0
    elif np.greater(y3, b):
        y3 = b

    row_idx = np.array([0, window_size])
    col_idx = np.array([0, window_size])
    interp_row = interp2d(row_idx, col_idx, [x0, x1, x2, x3])
    interp_col = interp2d(row_idx, col_idx, [y0, y1, y2, y3])

    row = np.arange(window_size)
    col = np.arange(window_size)
    coords_row = interp_row(row, col)
    coords_col = interp_col(row, col)

    X = map_coordinates(img, [coords_col, coords_row])
    if label is not None:
        y = map_coordinates(label, [coords_col, coords_row], order=0)
        return X, y
    return X, None


def get_max_nb_cls(dir_path: str):
    '''analyze the training folder and give the maximum number of classes'''
    assert os.path.isdir(dir_path), 'looking for a string of directory path'
    rws, gts, missing = check_raw_gt_pair(dir_path)
    if len(missing) != 0:
        raise Exception('Found missing gt for the following images: {}'.format(missing))
    all_cls = []
    for gt in gts:
        cls = np.unique(load_img(gt))
        for i in cls:
            if i not in all_cls:
                all_cls.append(i)
    max_nb_cls = len(all_cls)
    logger.info('In the traindata set folder, all cls: {}, max nb classes: {}'.format(all_cls, max_nb_cls))
    return all_cls, max_nb_cls

