import tensorflow as tf
import h5py
import numpy as np
import multiprocessing as mp
from PIL import Image
from util import check_N_mkdir, clean
from itertools import repeat
from input import _inverse_one_hot

import logging
import log
logger = log.setup_custom_logger('root')
logger.setLevel(logging.DEBUG)


def _h5Writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId, mode='onefile'):
    patch_shape = (patch_size, patch_size)
    # fill last .h5
    if mode == 'h5s':
        _h5s_writer(X_patches, y_patches, patch_shape, id_length, rest, outdir, patch_size, batch_size, maxId)
    elif mode == 'h5':
        _h5_writer(X_patches, y_patches, patch_shape, outdir, patch_size)
    elif mode == 'csvs':
        _csvs_writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId)
    elif mode == 'tfrecord':
        raise NotImplementedError("tfrecords part hasn't been implemented yet")
    else:
        raise ValueError("Please choose a mode from h5, csv or tfrecord!")


def _h5Writer_V2(X_patches, y_patches, outdir, patch_size):
    import os
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if not os.path.exists('{}{}'.format(outdir, patch_size)):
        os.mkdir('{}{}'.format(outdir, patch_size))

    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(_writer_V2, ((X_patches[i], y_patches[i], outdir, i, patch_size) for i in range(X_patches.shape[0])))


def _writer_V2(X, y, outdir, name, patch_size):
    with h5py.File('{}{}/{}.h5'.format(outdir, patch_size, name), 'w') as f:
        f.create_dataset('X', (patch_size, patch_size), dtype='float32', data=X)
        f.create_dataset('y', (patch_size, patch_size), dtype='float32', data=y)


def _h5Writer_V3(img_ID, w_ids, h_ids, in_path, patch_size, stride, outdir):
    assert isinstance(img_ID, int), 'Param ID should be interger'
    assert isinstance(stride, int), 'Stride should be interger'
    assert isinstance(w_ids, np.ndarray), 'Param ID should be np array'
    assert isinstance(h_ids, np.ndarray), 'Param ID should be np array'
    assert isinstance(in_path, str), 'Param ID should be np array'
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.starmap(_writer_V3, ((ID, xid, yid, _in_path, _outdir, stride, _patch_size)
                                  for ID, xid, yid, _in_path, _outdir, _patch_size, stride
                                  in zip(repeat(img_ID), w_ids, h_ids, repeat(in_path), repeat(outdir), repeat(patch_size), repeat(stride))))


def _writer_V3(img_ID, x_id, y_id, in_path, outdir, stride, patch_size):
    logger.debug(mp.current_process())
    X = np.asarray(Image.open(in_path + '.tif', 'r'))[x_id * stride: x_id * stride + patch_size, y_id * stride: y_id * stride + patch_size]
    y = np.asarray(Image.open(in_path + '_label.tif', 'r'))[x_id * stride: x_id * stride + patch_size, y_id * stride: y_id * stride + patch_size]
    with h5py.File('{}/{}_{}_{}.h5'.format(outdir, img_ID, x_id, y_id), 'w') as f:
        f.create_dataset('X', (patch_size, patch_size), dtype='float32', data=X)
        f.create_dataset('y', (patch_size, patch_size), dtype='float32', data=y)


def _csvs_writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId,):
    print('This will generate {} .csv in /proc/ directory'.format(id_length))
    if rest > 0:
        with open(outdir + 'X{}_{}_{}.csv'.format(patch_size, batch_size, maxId), 'ab') as f:
            np.savetxt(f, X_patches[:rest].ravel(), delimiter=',')  #csv can only save 1d or 2d array, we reshape on reading
        with open(outdir + 'y{}_{}_{}.csv'.format(patch_size, batch_size, maxId), 'a') as f:
            np.savetxt(f, y_patches[:rest].ravel(), delimiter=',')

    else:
        # then create new .h5
        for id in np.nditer(np.linspace(maxId, maxId + id_length, id_length, dtype='int')):
            try:
                start = rest + batch_size * (id - maxId)
                end = rest + batch_size * (id - maxId + 1)
                with open(outdir + 'X{}_{}_{}.csv'.format(patch_size, batch_size, id), 'wb') as f:
                    np.savetxt(f, X_patches[start:end, ].ravel(), delimiter=',')
                with open(outdir + 'y{}_{}_{}.csv'.format(patch_size, batch_size, id), 'wb') as f:
                    np.savetxt(f, y_patches[start:end, ].ravel(), delimiter=',')
            except Exception as e:
                print(e)
                # if the last one can't complete the whole .h5 file
                mod = (X_patches.shape[0] - rest) % batch_size
                with open(outdir + 'X{}_{}_{}.csv'.format(patch_size, batch_size, id), 'wb') as f:
                    np.savetxt(f, X_patches[-mod:].ravel(), delimiter=',')
                with open(outdir + 'y{}_{}_{}.csv'.format(patch_size, batch_size, id), 'wb') as f:
                    np.savetxt(f, y_patches[-mod:].ravel(), delimiter=',')


def _h5s_writer(X_patches, y_patches, patch_shape, id_length, rest, outdir, patch_size, batch_size, maxId,):
    print('This will generate {} .h5 in /proc/ directory'.format(id_length))
    if rest > 0:
        if len(X_patches.shape[0]) < rest:
            with h5py.File(outdir + '{}_{}_{}.h5'.format(patch_size, batch_size, maxId), 'a') as f:
                f['X'].resize(f['X'].shape[0] + rest, axis=0)
                f['y'].resize(f['y'].shape[0] + rest, axis=0)
                f['X'][-rest:] = X_patches[:rest]
                f['y'][-rest:] = y_patches[:rest]
        else:
            with h5py.File(outdir + '{}_{}_{}.h5'.format(patch_size, batch_size, maxId), 'a') as f:
                f['X'][-rest:] = X_patches[:rest]
                f['y'][-rest:] = y_patches[:rest]
    else:
        # then create new .h5
        for id in np.nditer(np.linspace(maxId, maxId + id_length, id_length, dtype='int')):
            try:
                with h5py.File(outdir + '{}_{}_{}.h5'.format(patch_size, batch_size, id), 'w') as f:
                    start = rest + batch_size * (id - maxId)
                    end = rest + batch_size * (id - maxId) + batch_size

                    X = f.create_dataset('X', (batch_size, *patch_shape),
                                         maxshape=(None, *patch_shape),
                                         dtype='float32')
                    X[:] = X_patches[start:end, ]
                    y = f.create_dataset('y', (batch_size, *patch_shape),
                                         maxshape=(None, *patch_shape),
                                         dtype='int8')
                    y[:] = y_patches[start:end, ]
            except:
                # if the last one can't complete the whole .h5 file
                with h5py.File(outdir + '{}_{}_{}.h5'.format(patch_size, batch_size, id), 'w') as f:
                    mod = (X_patches.shape[0] - rest) % batch_size
                    X = f.create_dataset('X', (batch_size, *patch_shape),
                                         maxshape=(None, *patch_shape),
                                         dtype='float32')
                    X[mod:] = X_patches[-mod:]
                    y = f.create_dataset('y', (batch_size, *patch_shape),
                                         maxshape=(None, *patch_shape),
                                         dtype='int8')
                    y[mod:] = y_patches[-mod:]


def _h5_writer(X_patches, y_patches, patch_shape, outdir, patch_size):
    try:
        with h5py.File(outdir + '{}.h5'.format(patch_size), 'a') as f:
            f['X'].resize(f['X'].shape[0] + X_patches.shape[0], axis=0)
            f['y'].resize(f['y'].shape[0] + y_patches.shape[0], axis=0)
            f['X'][-X_patches.shape[0]:] = X_patches[:X_patches.shape[0]]
            f['y'][-y_patches.shape[0]:] = y_patches[:y_patches.shape[0]]
        print('\n***Appended patches in .h5')

    except:
        with h5py.File(outdir + '{}.h5'.format(patch_size), 'w') as f:
            X = f.create_dataset('X', (X_patches.shape[0], *patch_shape),
                                 maxshape=(None, *patch_shape),
                                 dtype='float32')
            X[:] = X_patches[:X_patches.shape[0], ]
            y = f.create_dataset('y', (y_patches.shape[0], *patch_shape),
                                 maxshape=(None, *patch_shape),
                                 dtype='int8')
            y[:] = y_patches[:y_patches.shape[0], ]
        print('\n***Created new .h5')


def _resultWriter(tensor, layer_name='', path=None):
    '''
    tensor: images(numpy array or list of image) to save of (Height, Width, nth_Conv)
    path: path(string)
    layer_name: name of the layer(string)
    '''
    # mkdir
    check_N_mkdir(path + layer_name)

    # for writting inference partial rlt
    if isinstance(tensor, list):
        for i, elt in enumerate(tensor):
            logger.debug('layer_name: {}'.format(layer_name))
            if (True in np.isnan(elt)) or (True in np.isinf(elt)):
                try:
                    elt = clean(elt)
                except Exception as e:
                    logger.debug(e)
                    logger.warning('Clean function cannot clean the nan value from the layer {}! '.format(layer_name))
                    break

            # scope: conv p_infer: (ps, ps, nc)
            if elt.ndim == 3:
                # note: save only the first
                for j in range(elt.shape[-1]):
                    Image.fromarray(np.asarray(elt[:, :, j])).save(path + '{}/{}.tif'.format(layer_name, j))

            # scope: images
            elif elt.ndim == 2:
                Image.fromarray(np.asarray(elt)).save(path + '{}/{}.tif'.format(layer_name, i))

            # scope: dnn p_infer: (nb_post neuron)
            elif elt.ndim == 1:
                # reshape the 1D array
                ceil = int(np.ceil(np.sqrt(elt.size)))
                tmp = np.zeros((ceil ** 2), np.float32).ravel()
                tmp[:elt.size] = elt
                tmp = tmp.reshape(ceil, ceil)

                Image.fromarray(tmp).save(path + '{}/dnn.tif'.format(layer_name))

            elif elt.ndim == 4:
                elt = elt.squeeze()
                for j in range(elt.shape[0]):
                    Image.fromarray(np.asarray(elt[j])).save(path + '{}/{}.tif'.format(layer_name, j))
    else:
        #  treat dnn weights
        if tensor.ndim == 1:
            # reshape the 1D array
            ceil = int(np.ceil(np.sqrt(tensor.size)))
            tmp = np.zeros((ceil ** 2), np.float32).ravel()
            tmp[:tensor.size] = tensor
            tmp = tmp.reshape(ceil, ceil)

            Image.fromarray(tmp).save(path + '{}/dnn.tif'.format(layer_name))

        #  scope: not I_1_hoted tensor: (B, H, W, C)
        elif tensor.ndim == 4:
            tensor = np.squeeze(tensor.astype(np.float32))
            if 'diff' in layer_name or 'logit' in layer_name:
                for i in range(tensor.shape[0]):
                    Image.fromarray(tensor[i]).save(path + '{}/{}.tif'.format(layer_name, i))

        #  for cnn ndim=3
        elif tensor.ndim == 3:
            for i in range(tensor.shape[2]):
                Image.fromarray(np.asarray(tensor[:, :, i], dtype=np.float)).save(path + '{}/{}.tif'.format(layer_name, i))

        else:
            logger.warn('Not implement writer for this kind of data in layer :{}'.format(layer_name))
            raise NotImplementedError('Ouppss {}'.format(layer_name))


def _weighttWriter(tensor, layer_name='', path=None):
    '''
    tensor: images(numpy array or list of image) to save of (Height, Width, nth_Conv)
    path: path(string)
    layer_name: name of the layer(string)
    '''
    # mkdir
    check_N_mkdir(path + layer_name)

    # for writting inference partial rlt
    if isinstance(tensor, list):
        for i, elt in enumerate(tensor):
            logger.debug('layer_name: {}'.format(layer_name))
            if (True in np.isnan(elt)) or (True in np.isinf(elt)):
                try:
                    elt = clean(elt)
                except Exception as e:
                    logger.debug(e)
                    logger.warning('Clean function cannot clean the nan value from the layer {}! '.format(layer_name))
                    break

            # scope: conv p_infer: (ps, ps, nc)
            if elt.ndim == 3:
                # note: save only the first
                for j in range(elt.shape[-1]):
                    Image.fromarray(np.asarray(elt[:, :, j])).save(path + '{}/{}.tif'.format(layer_name, j))

            elif elt.ndim == 2:
                Image.fromarray(np.asarray(elt)).save(path + '{}/{}.tif'.format(layer_name, i))

            # scope: dnn p_infer: (nb_post neuron)
            elif elt.ndim == 1:
                # reshape the 1D array
                ceil = int(np.ceil(np.sqrt(elt.size)))
                tmp = np.zeros((ceil ** 2), np.float32).ravel()
                tmp[:elt.size] = elt
                tmp = tmp.reshape(ceil, ceil)

                Image.fromarray(tmp).save(path + '{}/dnn.tif'.format(layer_name))

    else:
        #  treat dnn weights
        if tensor.ndim == 1:
            # reshape the 1D array
            ceil = int(np.ceil(np.sqrt(tensor.size)))
            tmp = np.zeros((ceil ** 2), np.float32).ravel()
            tmp[:tensor.size] = tensor
            tmp = tmp.reshape(ceil, ceil)

            Image.fromarray(tmp).save(path + '{}/dnn.tif'.format(layer_name))

        #  scope: not I_1_hoted tensor: (B, H, W, C)
        elif tensor.ndim == 4:
            tensor = np.squeeze(tensor.astype(np.float32))
            if 'diff' in layer_name:
                for i in range(tensor.shape[0]):
                    Image.fromarray(tensor[i]).save(path + '{}/{}.tif'.format(layer_name, i))

        #  for cnn ndim=3
        elif tensor.ndim == 3:
            for i in range(tensor.shape[2]):
                Image.fromarray(np.asarray(tensor[:, :, i], dtype=np.float)).save(path + '{}/{}.tif'.format(layer_name, i))

        else:
            logger.warn('Not implement writer for this kind of data in layer :{}'.format(layer_name))
            raise NotImplementedError('Ouppss {}'.format(layer_name))
