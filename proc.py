import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import os
import h5py
from writer import _h5Writer, _tfrecordWriter, _h5Writer_V2
from reader import _tifReader




def preprocess(indir, stride, patch_size, mode='h5', shuffle=True, evaluate=True, traintest_split_rate=0.9):
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
                         y_patches[:np.int(X_patches.shape[0] * traintest_split_rate)],
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




def _shuffle(tensor_a, tensor_b, random_state=42):
    # shuffle two tensors in unison
    idx = np.random.permutation(tensor_a.shape[0]) #artifacts
    return tensor_a[idx], tensor_b[idx]



def _stride(tensor, stride, patch_size):
    p_h = (tensor.shape[0] - patch_size) // stride + 1
    p_w = (tensor.shape[1] - patch_size) // stride + 1
    # (4bytes * step * dim0, 4bytes * step, 4bytes * dim0, 4bytes)
    # stride the tensor
    _strides = tuple([i * stride for i in tensor.strides]) + tuple(tensor.strides)
    patches = as_strided(tensor, shape=(p_h, p_w, patch_size, patch_size), strides=_strides)\
        .reshape(-1, patch_size, patch_size)
    return patches


def _idParser(dir, patch_size, batch_size, mode='h5'):
    l_f = []
    max_id = 0
    # check if the .h5 with the same patch_size and batch_size exist
    for dirpath, _, fnames in os.walk(dir):
        for fname in fnames:
            if fname.split('_')[0] == patch_size and fname.split('_')[1] == batch_size and fname.endswith(mode):
                l_f.append(os.path.abspath(os.path.join(dirpath, fname)))
                max_id = max(max_id, int(fname.split('_')[2]))

    if mode == 'h5':
        try:
            with h5py.File(dir + '{}.'.format(patch_size) + mode, 'r') as f:
                rest = batch_size - f['X'].shape[0]
                return max_id, rest
        except:
            return 0, 0
    elif mode == 'csv':
        try:
            with open(dir + '{}_{}_{}.csv'.format(patch_size, batch_size, max_id) + mode, 'r') as f:
                rest = batch_size - f['X'].shape[0]
                return max_id, rest
        except:
            return 0, 0
    elif mode == 'tfrecord':
        raise NotImplementedError('tfrecord has not been implemented yet')




