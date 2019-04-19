import tensorflow as tf
import numpy as np
from tqdm import tqdm
import h5py
import os

totest = './proc/test/72/0.h5'

graph = tf.Graph()
with graph.as_default():
    with tf.Session() as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING], './dummy/savedmodel'
        )