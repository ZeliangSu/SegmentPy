import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import numpy as np
import os
from math import nan
from PIL import Image

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)  #changeHere: debug level


def print_nodes_name(graph):
    """
    input:
    -------
        graph: (tf.Graph() or tf.GraphDef()) graph in which prints only the nodes' name
    return:
    -------
        None
    """
    if isinstance(graph, graph_pb2.GraphDef):
        for n in graph.node:
            logger.info(n.name)
    else:
        for n in graph.as_graph_def().node:
            logger.info(n.name)


def print_nodes_name_shape(graph):
    """
    input:
    -------
        graph: (tf.Graph()) or tf.GraphDef()) graph in which prints the nodes' name and their shapes
    return:
    -------
        None
    """
    # fixme: enlarge to GraphDef
    if isinstance(graph, graph_pb2.GraphDef):
        # convert GraphDef to Graph
        graph = tf.import_graph_def(graph)

    for i in graph.get_operations():
        if len(i.outputs) is not 0:  #eliminate nodes like 'initializer' without tensor output
            for j in i.outputs:
                logger.info('{}: {}'.format(i.name, j.get_shape()))


def get_all_trainable_variables(metagraph_path):
    """
    input:
    -------
        metagraph_path: (string) indicate the path to find the metagraph of ckpt
    return:
    -------
        wn: (list of string) list of names of weights for all convolution and deconvolution layers
        bn: (list of string) list of names of bias for all convolution and deconvolution layers
        ws: (list of np.ndarray) list of weight matrices for all convolution and deconvolution layers
        bs: (list of np.ndarray) list of bias matrices for all convolution and deconvolution layers
        dnn_wn: (list of string) list of names of weights for all fully connected layers
        dnn_bn: (list of string) list of names of bias for all fully connected layers
        dnn_ws: (list of np.ndarray) list of weight matrices for all fully connected layers
        dnn_bs:(list of np.ndarray) list of bias matrices for all fully connected layers
    """
    tf.reset_default_graph()
    restorer = tf.train.import_meta_graph(
        metagraph_path + '.meta',
        clear_devices=True
    )

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        restorer.restore(sess, metagraph_path)
        all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        wn = [v.name for v in all_vars if v.name.endswith('w:0') and not v.name.startswith('dnn')]
        bn = [v.name for v in all_vars if v.name.endswith('b:0') and not v.name.startswith('dnn')]
        dnn_wn = [v.name for v in all_vars if v.name.endswith('w:0') and v.name.startswith('dnn')]
        dnn_bn = [v.name for v in all_vars if v.name.endswith('b:0') and v.name.startswith('dnn')]
        ws = [sess.run(v) for v in all_vars if v.name.endswith('w:0') and not v.name.startswith('dnn')]
        bs = [sess.run(v) for v in all_vars if v.name.endswith('b:0') and not v.name.startswith('dnn')]
        dnn_ws = [sess.run(v) for v in all_vars if v.name.endswith('w:0') and v.name.startswith('dnn')]
        dnn_bs = [sess.run(v) for v in all_vars if v.name.endswith('b:0') and v.name.startswith('dnn')]

    return wn, bn, ws, bs, dnn_wn, dnn_bn, dnn_ws, dnn_bs


def check_N_mkdir(path_to_dir):
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir, exist_ok=True)


def clean(array, clean_zeros=False):
    assert isinstance(array, np.ndarray)
    array[np.where(array == np.nan)] = 1e-10
    array[np.where(array == nan)] = 1e-10
    if clean_zeros:
        array[np.where(array == 0)] = 1e-10
    array[np.where(array == np.inf)] = 1e9
    array[np.where(array == -np.inf)] = -1e9
    return array


class plot_input_logit_label_diff():
    def __init__(self):
        self.input = None
        self.logit = None
        self.label = None
        self.diff = None
        self.out_path = None

    def add_input(self, input):
        assert isinstance(input, np.ndarray)
        self.input = np.squeeze(input)

    def add_logit(self, logit):
        assert isinstance(logit, np.ndarray)
        self.logit = np.squeeze(logit)

    def add_label(self, label):
        assert isinstance(label, np.ndarray)
        self.label = np.squeeze(label)

    def add_diff(self, diff):
        assert isinstance(diff, np.ndarray)
        self.diff = np.squeeze(diff)

    def plot(self, out_path):
        self.out_path = out_path
        assert self.input.shape == self.logit.shape == self.diff.shape == self.label.shape, 'Shapes of in/out/lab/diff not match, or lack of one element'
        assert self.out_path is not None, 'Need to indicate a out path.'
        if self.input.ndim == 2:
            final = np.zeros((self.input.shape[0] * 2 + 10, self.input.shape[1] * 2 + 10))
            final[:self.input.shape[0], :self.input.shape[1]] = self.input
            final[:self.input.shape[0], self.input.shape[1] + 10:] = self.logit
            final[self.input.shape[0] + 10:, self.input.shape[1]:] = self.label
            final[self.input.shape[0] + 10:, :self.input.shape[1] + 10] = self.diff
            Image.fromarray(final).save(self.out_path)

        elif self.input.ndim == 3:
            # should be shape of (id, H, W)
            for i in range(self.input.shape[0]):
                final = np.zeros((self.input[i].shape[0] * 2 + 10, self.input[i].shape[1] * 2 + 10))
                final[:self.input.shape[1], :self.input.shape[2]] = self.input[i]
                final[:self.input.shape[1], self.input.shape[2] + 10:] = self.logit[i]
                final[self.input.shape[1] + 10:, :self.input.shape[2]] = self.label[i]
                final[self.input.shape[1] + 10:, self.input.shape[2] + 10:] = self.diff[i]
                Image.fromarray(final).save(self.out_path)
                Image.fromarray(final).save(self.out_path.replace('.tif', '{}.tif'.format(i)))

        else:
            raise ValueError('Expected an image or a stack of image')


def get_list_fnames(directory):
    l_fname = os.listdir(directory)
    return [directory + fname for fname in l_fname]


def exponential_decay(total_step, initial_lr, k=0.1):
    steps = np.linspace(0, total_step, total_step)
    lr_np = initial_lr * np.exp(- k * steps)
    return lr_np.astype(float)


def ramp_decay(total_step, nb_batch, initial_lr, k=0.5, period=1):
    epochs = np.linspace(0, total_step, total_step) // int(nb_batch * period)  #e.g. period = 3: every 3 epochs decrease the lr
    lr_np = initial_lr * pow(k, epochs)
    return lr_np.astype(float)  # the placeholder of lr is float32 mysterious using float get better accuracy


class check_identical():
    pass


class ckpt():
    def here(self):
        print("I'm here")

