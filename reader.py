from PIL import Image
import os
import numpy as np
import tensorflow as tf


def _tifReader(dir):
    l_X = []
    for dirpath, _, fnames in os.walk(dir):
        for fname in fnames:
            if 'label' not in fname:
                l_X.append(os.path.abspath(os.path.join(dirpath, fname)))
    l_X = sorted(l_X)

    # collect img
    X_stack = []
    y_stack = []
    shapes = []
    for f_X in l_X:
        X_img = np.asarray(Image.open(f_X))
        try:
            y_img = np.asarray(Image.open(f_X.split('.')[-2] + '_label.tif'))
        except:
            print('cannot find _label.tif but continue')
            y_img = np.empty(X_img.shape)


        # check dimensions
        if X_img.shape != y_img.shape:
            raise ValueError('shape of image output {} is different from input {}'.format(y_img.shape, X_img.shape))

        X_stack.append(X_img)
        y_stack.append(y_img)
        shapes.append(X_img.shape)
    return X_stack, y_stack, shapes #lists


def tfrecordReader(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(
        serialized_example,
        features={
            'X': tf.FixedLenFeature([], tf.string),
            'y': tf.FixedLenFeature([], tf.string),
        })

    X = tf.decode_raw(img_features['X'], tf.float32)
    y = tf.decode_raw(img_features['y'], tf.float32)
    return X, y

