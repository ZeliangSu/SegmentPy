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

    # stack params
    mapping = []
    step = np.asarray(get_sum(accumulator, l_grad_tag[0])[1])
    for grad in l_grad_tag:
        mapping.append(np.asarray(get_sum(accumulator, grad)[0]))
    mapping = np.stack(mapping, axis=1)
    Image.fromarray(mapping).save('./dummy/mapping.tif')


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
    # ac_tn, ac_tt, ls_tn, ls_tt = lr_curve_extractor('/Users/zeliangsu/Desktop/event')
    gradient_extractor('/Users/zeliangsu/Desktop/event/data/train')
    print('ha')
