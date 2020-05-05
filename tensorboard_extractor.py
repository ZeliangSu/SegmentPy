from tensorboard.backend.event_processing import event_accumulator
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re

import logging
import log
logger = log.setup_custom_logger('root')
logger.setLevel(logging.DEBUG)


def gradient_extractor(event_dir: str):
    accumulator = event_accumulator.EventAccumulator(event_dir,
                                                     size_guidance={
                                                         event_accumulator.SCALARS: 0,
                                                         event_accumulator.HISTOGRAMS: 0,
                                                     })
    accumulator.Reload()
    tags = accumulator.Tags()
    l_grad_tag = []
    for param_name in tags['histograms']:
        if 'grad' in param_name:
            l_grad_tag.append(param_name)
    print(l_grad_tag)

    # stack params
    mapping = []
    gamma = []
    beta = []
    w = []
    layer = {}
    step = np.asarray(get_sum(accumulator, l_grad_tag[0])[1])
    for grad in l_grad_tag:
        mapping.append(np.asarray(get_sum(accumulator, grad)[0]))
        if 'gamma' in grad:
            gamma.append(np.asarray(get_sum(accumulator, grad)[0]))
        elif 'beta' in grad:
            beta.append(np.asarray(get_sum(accumulator, grad)[0]))
        elif 'w_0' in grad:
            w.append(np.asarray(get_sum(accumulator, grad)[0]))
        try:
            layer_name = re.search('conv(\d+)', grad).group(1)
            try:
                layer[layer_name].append(np.asarray(get_sum(accumulator, grad)[0]))
            except Exception as e:
                print(e)
                layer[layer_name] = [np.asarray(get_sum(accumulator, grad)[0])]
        except Exception as e:
            print(grad)
            pass

    layer_mapping = []
    for k, v in layer.items():
        # take absolute value
        layer_mapping.append(np.sum(np.abs(elt) for elt in v))
    layer_mapping = np.stack(layer_mapping).transpose()
    mapping = np.stack(mapping, axis=1)
    gamma = np.stack(gamma, axis=1)
    beta = np.stack(beta, axis=1)
    w = np.stack(w, axis=1)

    # fold N times then sum
    N = 50
    M = 1

    # solution: repeat N times is easier
    layer_mapping = np.repeat(np.repeat(layer_mapping, N, axis=1), M, axis=0)
    mapping = np.repeat(np.repeat(mapping, N, axis=1), M, axis=0)
    gamma = np.repeat(np.repeat(gamma, N, axis=1), M, axis=0)
    beta = np.repeat(np.repeat(beta, N, axis=1), M, axis=0)
    w = np.repeat(np.repeat(w, N, axis=1), M, axis=0)

    # save gradient mappings
    Image.fromarray(layer_mapping).save('./dummy/each_layer_mapping.tif')
    Image.fromarray(mapping).save('./dummy/all_param_mapping.tif')
    Image.fromarray(gamma).save('./dummy/gamma.tif')
    Image.fromarray(beta).save('./dummy/beta.tif')
    Image.fromarray(w).save('./dummy/w.tif')


def get_sum(accumulator, param_name):
    l_event = accumulator.Histograms(param_name)
    l_sum = []
    l_step = []
    for event in l_event:
        l_sum.append(event.histogram_value.sum)
        l_step.append(event.step)
    return l_sum, l_step


def lr_curve_extractor(event_dir: str):
    accumulator = event_accumulator.EventAccumulator(event_dir,
                                               size_guidance={
                                                   event_accumulator.SCALARS: 0,
                                                   event_accumulator.HISTOGRAMS: 0,
                                               })
    accumulator.Reload()

    try:
        acc_train = pd.DataFrame(accumulator.Scalars('train_metrics/accuracy')).drop(columns=['wall_time'])
    except Exception as e:
        logger.debug(e)
        acc_train = None

    try:
        acc_test = pd.DataFrame(accumulator.Scalars('test_metrics/accuracy')).drop(columns=['wall_time'])
    except Exception as e:
        logger.debug(e)
        acc_test = None

    try:
        lss_train = pd.DataFrame(accumulator.Scalars('train_metrics/loss')).drop(columns=['wall_time'])
    except Exception as e:
        logger.debug(e)
        lss_train = None

    try:
        lss_test = pd.DataFrame(accumulator.Scalars('test_metrics/loss')).drop(columns=['wall_time'])
    except Exception as e:
        logger.debug(e)
        lss_test = None

    return acc_train, acc_test, lss_train, lss_test


if __name__ == '__main__':
    """[libprotobuf ERROR external/protobuf_archive/src/google/protobuf/descriptor_database.cc:334] Invalid file descriptor data passed to EncodedDescriptorDatabase::Add().
[libprotobuf FATAL external/protobuf_archive/src/google/protobuf/descriptor.cc:1370] CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size): 
libc++abi.dylib: terminating with uncaught exception of type google::protobuf::FatalException: CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size): """

    """solution: conda install protobuf=3.8 """

    # ac_tn, ac_tt, ls_tn, ls_tt = lr_curve_extractor('/Users/zeliangsu/Desktop/event')
    gradient_extractor('/Users/zeliangsu/Desktop/event/data/LRCS7/train/')
    print('ha')
