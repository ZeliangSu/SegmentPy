import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import os
import h5py
from itertools import repeat
from writer import _h5Writer_V2, _h5Writer_V3
from reader import _tifReader
from util import check_N_mkdir


def preprocess(indir, stride, patch_size, mode='h5', shuffle=True, evaluate=True, traintest_split_rate=0.9):
    """
    input:
    -------
        indir: (string)
        stride: (int) step of pixel
        patch_size: (int) height and width of
        mode: (string) file type of to save the preprocessed images. #TODO: .csv .tiff
        shuffle: (boolean) if True, preprocessed images will be shuffled before saving
        evaluate: (boolean) if True, preprocessed images will be saved into two directories for training set and test set
        traintest_split_rate: (float) the ratio for splitting trainning/test set
    return:
    -------
        None
    """
    # todo: can simplify to not loading in RAM
    # import data
    X_stack, y_stack, _ = _tifReader(indir)
    outdir = './proc/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    X_patches = _stride(X_stack[0], stride, patch_size)
    y_patches = _stride(y_stack[0], stride, patch_size)

    # extract patches
    for i in range(1, len(X_stack) - 1):
        X_patches = np.vstack((X_patches, _stride(X_stack[i], stride, patch_size)))
    for i in range(1, len(y_stack) - 1):
        y_patches = np.vstack((y_patches, _stride(y_stack[i], stride, patch_size)))

    assert X_patches.shape[0] == y_patches.shape[0], 'numbers of raw image: {} and label image: {} are different'.format(X_patches.shape[0], y_patches.shape[0])

    # shuffle
    if shuffle:
        X_patches, y_patches = _shuffle(X_patches, y_patches)

    if evaluate:
        if mode == 'h5':
            _h5Writer_V2(X_patches[:np.int(X_patches.shape[0] * traintest_split_rate)],
                         y_patches[:np.int(X_patches.shape[0] * traintest_split_rate)],
                         outdir + 'train/', patch_size)
            _h5Writer_V2(X_patches[np.int(X_patches.shape[0] * traintest_split_rate):],
                         y_patches[np.int(X_patches.shape[0] * traintest_split_rate):],
                         outdir + 'test/', patch_size)
        elif mode == 'csv':
            raise NotImplementedError
        else:
            raise NotImplementedError

    else:
        if mode == 'h5':
            _h5Writer_V2(X_patches, y_patches, outdir, patch_size)
        elif mode == 'csv':
            raise NotImplementedError
        else:
            raise NotImplementedError


def preprocess_V2(indir, stride, patch_size, traintest_split_rate=0.9, shuffle=False):
    """
    input:
    -------
        indir: (string)
        stride: (int) step of pixel
        patch_size: (int) height and width of
        mode: (string) file type of to save the preprocessed images. #TODO: .csv .tiff
        shuffle: (boolean) if True, preprocessed images will be shuffled before saving
        evaluate: (boolean) if True, preprocessed images will be saved into two directories for training set and test set
        traintest_split_rate: (float) the ratio for splitting trainning/test set
    return:
    -------
        None
    """
    train_outdir = './proc/train/{}/'.format(patch_size)
    test_outdir = './proc/test/{}/'.format(patch_size)
    check_N_mkdir(train_outdir)
    check_N_mkdir(test_outdir)

    # import data
    X_stack, y_stack, list_shapes = _tifReader(indir)
    assert (len(X_stack) == len(y_stack) == len(list_shapes)), 'Number of img, label, and their shapes are not equal!'

    # get ID, nb_X, nb_y
    list_ID = [id for id in range(len(list_shapes))]
    list_nb_w = [(np.asarray(X_stack[i]).shape[0] - patch_size) // stride + 1 for i in range(len(X_stack))]
    list_nb_h = [(np.asarray(y_stack[i]).shape[0] - patch_size) // stride + 1 for i in range(len(y_stack))]

    # make ID grid then pick e.g. 90% for train and 10% for test
    train_id_dict = {}
    test_id_dict = {}
    for ID, nb_w, nb_h in zip(list_ID, list_nb_w, list_nb_h):
        # build a x-y grid
        xid, yid = np.meshgrid(np.arange(nb_w), np.arange(nb_h))  # xv, yv same shape
        xid, yid = np.reshape(xid, (-1)), np.reshape(yid, (-1))  # flatten xv, yv
        tmp = np.arange(xid.size)

        # choose 90% of the pixel
        random = np.random.choice(tmp, int(xid.size * traintest_split_rate), replace=False)

        tmp = np.zeros(xid.shape)
        tmp[random] = 1
        train_id_dict[ID] = [xid[np.where(tmp == 1)], yid[np.where(tmp == 1)]]
        test_id_dict[ID] = [xid[np.where(tmp == 0)], yid[np.where(tmp == 0)]]

    # X, y coords
    # train set
    for img_id, _indir, patch_size, outdir in zip(list_ID, repeat(indir), repeat(patch_size), repeat(train_outdir)):
        _h5Writer_V3(img_ID=img_id,
                     w_ids=train_id_dict[img_id][0],
                     h_ids=train_id_dict[img_id][1],
                     in_path=indir + str(img_id),
                     stride=stride,
                     patch_size=patch_size,
                     outdir=train_outdir)

    # test set
    for img_id in list_ID:
        _h5Writer_V3(img_ID=img_id,
                     w_ids=test_id_dict[img_id][0],
                     h_ids=test_id_dict[img_id][1],
                     in_path=indir + str(img_id),
                     patch_size=patch_size,
                     stride=stride,
                     outdir=test_outdir)


def _shuffle(tensor_a, tensor_b, random_state=42):
    """
    input:
    -------
        tensor_a: (np.ndarray) input tensor
        tensor_b: (np.ndarray) input tensor
    return:
    -------
        tensor_a: (np.ndarray) shuffled tensor_a at the same way as tensor_b
        tensor_b: (np.ndarray) shuffled tensor_b at the same way as tensor_a
    """
    # shuffle two tensors in unison
    np.random.seed(random_state)
    idx = np.random.permutation(tensor_a.shape[0]) #artifacts
    return tensor_a[idx], tensor_b[idx]


def _stride(tensor, stride, patch_size):
    """
    input:
    -------
        tensor: (np.ndarray) images to stride
        stride: (int) pixel step that the window of patch jump for sampling
        patch_size: (int) height and weight (here we assume the same) of the sampling image
    return:
    -------
        patches: (np.ndarray) strided and restacked patches
    """
    p_h = (tensor.shape[0] - patch_size) // stride + 1
    p_w = (tensor.shape[1] - patch_size) // stride + 1
    # (4bytes * step * dim0, 4bytes * step, 4bytes * dim0, 4bytes)
    # stride the tensor
    _strides = tuple([i * stride for i in tensor.strides]) + tuple(tensor.strides)
    patches = as_strided(tensor, shape=(p_h, p_w, patch_size, patch_size), strides=_strides)\
        .reshape((-1, patch_size, patch_size))
    return patches


def _idParser(directory, patch_size, batch_size, mode='h5'):
    """
    input:
    -------
        directory: (string) path to be parsed
        patch_size: (int) height and weight (here we assume the same)
        batch_size: (int) number of images per batch
        mode: (string) file type to be parsed
    return:
    -------
        None
    """
    l_f = []
    max_id = 0
    # check if the .h5 with the same patch_size and batch_size exist
    for dirpath, _, fnames in os.walk(directory):
        for fname in fnames:
            if fname.split('_')[0] == patch_size and fname.split('_')[1] == batch_size and fname.endswith(mode):
                l_f.append(os.path.abspath(os.path.join(dirpath, fname)))
                max_id = max(max_id, int(fname.split('_')[2]))

    if mode == 'h5':
        try:
            with h5py.File(directory + '{}.'.format(patch_size) + mode, 'r') as f:
                rest = batch_size - f['X'].shape[0]
                return max_id, rest
        except:
            return 0, 0
    elif mode == 'csv':
        try:
            with open(directory + '{}_{}_{}.csv'.format(patch_size, batch_size, max_id) + mode, 'r') as f:
                rest = batch_size - f['X'].shape[0]
                return max_id, rest
        except:
            return 0, 0
    elif mode == 'tfrecord':
        raise NotImplementedError('tfrecord has not been implemented yet')




