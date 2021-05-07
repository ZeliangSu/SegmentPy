import tensorflow as tf
from tensorflow.core.framework import graph_pb2
import numpy as np
import os
from math import nan
from PIL import Image
from skimage import exposure
import re
import shutil

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.WARNING)  #changeHere: debug level


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
        return [n.name for n in graph.node]
    else:
        for n in graph.as_graph_def().node:
            logger.info(n.name)
        return [n.name for n in graph.as_graph_def().node]


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
        metagraph_path if metagraph_path.endswith('.meta') else metagraph_path + '.meta',  # here with .meta
        clear_devices=True
    )

    with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
        restorer.restore(sess, metagraph_path.replace('.meta', '') if metagraph_path.endswith('.meta') else metagraph_path)  # here without .meta
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


def duplicate_event(path: str):
    assert path.endswith('/'), 'should give a dir'
    if not os.path.exists(path + 'event/'):
        shutil.copytree(path + 'train/', path + 'event/')
        for f in os.listdir(path + 'test/'):
            if os.path.exists(path + 'event/' + f):
                # e.g. re.sub('tfevents\.(\d+)(\.)t', r'\1(1)\2','events.out.tfevents.1591006736.tomoserver')
                # --> 'events.out.1591006736(1).omoserver'
                # --> should fill it
                shutil.copyfile(path + 'test/' + f, path + 'event/' + re.sub('(tfevents\.\d+)(\.)', r'\1(1)\2', f))
            else:
                shutil.copyfile(path + 'test/' + f, path + 'event/' + f)


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
        if input.shape[-1] == 10:
            self.input = input[:, :, 0]
        else:
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
        self.input = _minmaxscalar(self.input)
        self.logit = _minmaxscalar(self.logit)
        self.diff = _minmaxscalar(self.diff)
        self.label = _minmaxscalar(self.label)
        if self.input.ndim == 2:
            final = np.zeros((self.input.shape[0] * 2 + 10, self.input.shape[1] * 2 + 10))
            final[:self.input.shape[0], :self.input.shape[1]] = self.input
            final[:self.input.shape[0], self.input.shape[1] + 10:] = self.logit
            final[self.input.shape[0] + 10:, self.input.shape[1] + 10:] = self.label
            final[self.input.shape[0] + 10:, :self.input.shape[1]] = self.diff
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


class ckpt():
    def here(self):
        print("I'm here")


def list_ckpts(directory):
    fnames = os.listdir(directory)
    ckpts = []
    fns = []
    for fname in fnames:
        if fname.endswith('.meta'):
            ckpts.append(int(fname.split('step')[1].split('.')[0]))
            fns.append(directory + fname.replace('.meta', ''))
    ckpts = sorted(ckpts)
    fns = sorted(fns)
    print(ckpts)
    return ckpts, fns


def dimension_regulator(img, maxp_times=3):
    ''' some models constraint the i/o dimensions should be multiple of 8 (for 3 times maxpooling)'''
    # note: the following dimensions should be multiple of 8 if 3x Maxpooling
    multiple = 2 ** maxp_times
    # note: fiji (w:1572, h:1548) --> PIL --> np.shape(row: 1548, col: 1572)
    w, h = img.shape[1] % multiple, img.shape[0] % multiple
    a, b = img.shape[1] // multiple, img.shape[0] // multiple
    img = img[h // 2: h // 2 + b * multiple, w // 2: w // 2 + a * multiple]
    return img


def load_img(path):
    logger.info('path: %s' % path)
    img = np.asarray(Image.open(path))
    return img


def auto_contrast(img):
    img = exposure.equalize_hist(img)
    return img


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


def get_img_stack(img_dir:str, img_or_label:str):
    fns = [img_dir + f for f in os.listdir(img_dir)]
    imgs = []
    for p in fns:
        if img_or_label in ['input', 'img']:
            if re.search('(\d+)(\.tif)', p):
                imgs.append(dimension_regulator(load_img(p)))
        elif img_or_label == 'label':
            if re.search('(_label)(\.tif)', p):
                imgs.append(dimension_regulator(load_img(p)))
        else:
            raise ValueError('input, img, or label')
    imgs = np.stack(imgs, axis=0)
    return imgs


def read_pb(pb_path):
    # tf.reset_default_graph()
    with tf.gfile.GFile(pb_path, 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())
    return graph_def_optimized


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
    return X_stack, \
           y_stack, \
           shapes #lists


def boolean_string(s):
    if s in ['False', 'false']:
        return False
    else:
        return True


def check_raw_gt_pair(dir_path: str):
    assert os.path.isdir(dir_path), 'need a string of directory path'
    list_raw = []
    list_gt = []

    for rel in os.listdir(dir_path):
        if '_label.tif' in rel:
            list_gt.append(dir_path + rel)
        else:
            list_raw.append(dir_path + rel)
    if len(list_raw) >= len(list_gt):
        logger.debug('detected {} of raw images and {} of gt images'.format(len(list_raw), len(list_gt)))
    else:
        logger.warning('found more images of gt{} than raw {}'.format(len(list_gt), len(list_raw)))
    missing_gt = []
    for rw in list_raw:
        if rw.replace('.tif', '_label.tif') not in list_gt:
            missing_gt.append(dir_path + rw)
    return list_raw, list_gt, missing_gt


def get_latest_training_number(dir_path: str):
    assert os.path.isdir(dir_path), 'need a string of directory path'
    l_fn = os.listdir(dir_path)
    latest = 0
    if len(l_fn) > 0:
        for fn in l_fn:
            tmp = re.search('^(\d+)\_', fn)
            if tmp is not None:
                tmp = int(tmp.group(1))
                latest = max(tmp, latest)
    return latest

