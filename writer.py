import tensorflow as tf
import h5py
import numpy as np
import multiprocessing as mp

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
        #TODO:
        raise NotImplementedError("tfrecords part hasn't been implemented yet")
    elif mode == 'npy':
        #TODO:
        raise NotImplementedError("npy part hasn't been implemented yet")

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

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _tfrecordWriter(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId):
    #TODO: the tf.io.TFRecordWriter cannot do yet append in tensorflow V1.12, so can only write the multiple
    #TODO: of batch_size in .tfrecoord here (different from .h5)

    for id in np.nditer(np.linspace(maxId + 1, maxId + id_length, id_length - 1, dtype='int')):
        with tf.io.TFRecordWriter(outdir + '{}_{}_{}.tfrecord'.format(patch_size, batch_size, id)) as writer:
            start = rest + batch_size * (id - maxId)
            end = rest + batch_size * (id - maxId) + batch_size

            for i in np.nditer(np.linspace(start, end, batch_size)):
                # Create a feature
                feature = {
                    'image_raw': _bytes_feature(X_patches[i, ].tostring()),
                    'label': _bytes_feature(y_patches[i, ].tostring())
                }
                # Create an example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # Serialize to string and write on the file
                writer.write(example.SerializeToString())

