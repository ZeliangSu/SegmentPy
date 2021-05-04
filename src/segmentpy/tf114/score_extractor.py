from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
import os

from segmentpy.tf114.util import check_N_mkdir

import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger('root')
logger.setLevel(logging.DEBUG)


def gradient_extractor(event_dir: str, write_rlt=True):

    if not event_dir.endswith('train/'):
        _dir = os.path.join(event_dir, 'train/')
    else:
        _dir = event_dir
    accumulator = event_accumulator.EventAccumulator(_dir,
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
    logger.info(l_grad_tag)

    # stack params
    mapping = []
    gamma = []
    gamman = []
    beta = []
    betan = []
    w = []
    wn = []
    layer = {}
    step = np.asarray(get_sum(accumulator, l_grad_tag[0])[1])
    for grad in l_grad_tag:
        mapping.append(np.asarray(get_sum(accumulator, grad)[0]))
        if 'gamma' in grad:
            gamma.append(np.asarray(get_sum(accumulator, grad)[0]))
            gamman.append(grad)
        elif ('beta' in grad) or ('b_0' in grad):
            beta.append(np.asarray(get_sum(accumulator, grad)[0]))
            betan.append(grad)
        elif 'w_0' in grad:
            w.append(np.asarray(get_sum(accumulator, grad)[0]))
            wn.append(grad)
        try:
            layer_name = re.search('conv(\d+b?)', grad).group(1)
            try:
                layer[layer_name].append(np.asarray(get_sum(accumulator, grad)[0]))
            except Exception as e:
                logger.error(e)
                layer[layer_name] = [np.asarray(get_sum(accumulator, grad)[0])]
        except Exception as e:
            logger.error(grad)
            pass

    block_mapping = []
    layer_mapping = []
    for k, v in layer.items():
        # take absolute value
        layer_mapping.append(np.sum(np.abs(elt) for elt in v))
        if 'b' in k:
            block_mapping.append(np.sum(np.abs(elt) for elt in v))

    block_mapping = np.stack(block_mapping).transpose()
    layer_mapping = np.stack(layer_mapping).transpose()
    full_mapping = np.stack(mapping, axis=1)

    # take the absolute values of the gradients
    gamma = np.abs(np.stack(gamma, axis=1))
    beta = np.abs(np.stack(np.abs(beta), axis=1))
    w = np.abs(np.stack(np.abs(w), axis=1))

    # fold N times then sum
    N = 50
    M = 1

    # solution: repeat N times is easier
    block_mapping = np.repeat(np.repeat(block_mapping, N, axis=1), M, axis=0)
    layer_mapping = np.repeat(np.repeat(layer_mapping, N, axis=1), M, axis=0)
    full_mapping = np.repeat(np.repeat(full_mapping, N, axis=1), M, axis=0)
    gamma = np.repeat(np.repeat(gamma, N, axis=1), M, axis=0)
    beta = np.repeat(np.repeat(beta, N, axis=1), M, axis=0)
    w = np.repeat(np.repeat(w, N, axis=1), M, axis=0)

    if write_rlt:
        check_N_mkdir(event_dir + 'grad/')

        # save gradient mappings
        Image.fromarray(block_mapping).save(event_dir + 'grad/each_block_mapping.tif')
        Image.fromarray(layer_mapping).save(event_dir + 'grad/each_layer_mapping.tif')
        Image.fromarray(full_mapping).save(event_dir + 'grad/all_param_mapping.tif')
        Image.fromarray(gamma).save(event_dir + 'grad/gamma.tif')
        Image.fromarray(beta).save(event_dir + 'grad/beta.tif')
        Image.fromarray(w).save(event_dir + 'grad/w.tif')

        # save gradient as .csv
        np.savetxt(event_dir + "grad/each_layer_mapping.csv", block_mapping, delimiter=",")
        np.savetxt(event_dir + "grad/all_param_mapping.csv", full_mapping, delimiter=",")
        np.savetxt(event_dir + "grad/gamma.csv", gamma, delimiter=",")
        np.savetxt(event_dir + "grad/beta.csv", beta, delimiter=",")
        np.savetxt(event_dir + "grad/w.csv", w, delimiter=",")

    _gamma = {}
    _betaOrBias = {}
    _w = {}

    for i, n in enumerate(gamman):
        #gamma.shape: (step, N*nb_layer), gamman.shape: (nb_layer)
        _gamma[n] = gamma[::M, i * N]

    for i, n in enumerate(betan):
        _betaOrBias[n] = beta[::M, i * N]

    for i, n in enumerate(wn):
        _w[n] = w[::M, i * N]

    return block_mapping, full_mapping, _gamma, _betaOrBias, _w, step


def get_sum(accumulator, param_name):
    l_event = accumulator.Histograms(param_name)
    l_sum = []
    l_step = []
    for event in l_event:
        l_sum.append(event.histogram_value.sum)
        l_step.append(event.step)
    return l_sum, l_step


def lr_curve_extractor(event_dir: str):
    logger.info(event_dir)
    if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(event_dir)), 'curves', 'acc_test.csv')):
        logger.debug('found the accuracy summary .csv file')
        curvesDir = os.path.join(os.path.dirname(os.path.dirname(event_dir)), 'curves')
        return pd.read_csv(os.path.join(curvesDir, 'acc_train.csv')), \
               pd.read_csv(os.path.join(curvesDir, 'acc_test.csv')), \
               pd.read_csv(os.path.join(curvesDir, 'lss_train.csv')), \
               pd.read_csv(os.path.join(curvesDir, 'lss_test.csv')),

    else:
        logger.debug('cannot found the accuracy summary .csv file')
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


def df_to_csv(where: str, acc_train, acc_test, lss_train, lss_test):
    assert isinstance(acc_train, pd.DataFrame)
    assert isinstance(acc_test, pd.DataFrame)
    assert isinstance(lss_train, pd.DataFrame)
    assert isinstance(lss_test, pd.DataFrame)
    assert os.path.isdir(where)

    acc_train.to_csv(where + '/acc_train.csv')
    acc_test.to_csv(where + '/acc_test.csv')
    lss_train.to_csv(where + '/lss_train.csv')
    lss_test.to_csv(where + '/lss_test.csv')


def get_pd_lr_curves(pd_dir: str):
    assert os.path.isdir(pd_dir)
    acc_train = None
    acc_test = None
    lss_train = None
    lss_test = None
    try:
        if not os.path.exists(pd_dir+'curves/acc_train.csv'):
            # todo: this way of finding score is not accurate
            logger.warning('[SegmentPy]: could not find the lr curves .csv file')
        else:
            acc_train = pd.read_csv(pd_dir + '/acc_train.csv')
            acc_test = pd.read_csv(pd_dir + '/acc_test.csv')
            lss_train = pd.read_csv(pd_dir + '/lss_train.csv')
            lss_test = pd.read_csv(pd_dir + '/lss_test.csv')

    except Exception as e:
        logger.error(e)

    return acc_train, acc_test, lss_train, lss_test


def get_test_acc(log_dir: str):
    assert os.path.isdir(log_dir)
    acc = None
    try:
        if not os.path.exists(log_dir+'curves/test_score.csv'):
            acc = 0
            # todo: this way of finding score is not accurate
            logger.warning('[SegmentPy]: could not find the test score folder')
        else:
            acc = pd.read_csv(log_dir+'curves/test_score.csv', header=True).acc.mean()

    except Exception as e:
        logger.error(e)

    return acc


def extractor_wrapper(parent_dir: str):
    '''parent_dir: the training holder ends with /hour{}_gpu{}/'''
    assert os.path.isdir(parent_dir)
    acc_train = None
    acc_test = None
    lss_train = None
    lss_test = None
    if not os.path.exists(parent_dir):
        logger.warning('[SegmentPy]: did not found the folder to extract the lr curves')
    else:
        if os.path.exists(parent_dir+'curves/'):
            acc_train, acc_test, lss_train, lss_test = get_pd_lr_curves(parent_dir)
        elif os.path.exists(parent_dir+'train/'):
            acc_train, acc_test, lss_train, lss_test = lr_curve_extractor(parent_dir)
        else:
            logger.warning('[SegmentPy]: could not find original lr curves')
    return acc_train, acc_test, lss_train, lss_test


if __name__ == '__main__':
    """[libprotobuf ERROR external/protobuf_archive/src/google/protobuf/descriptor_database.cc:334] Invalid file descriptor data passed to EncodedDescriptorDatabase::Add().
[libprotobuf FATAL external/protobuf_archive/src/google/protobuf/descriptor.cc:1370] CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size): 
libc++abi.dylib: terminating with uncaught exception of type google::protobuf::FatalException: CHECK failed: GeneratedDatabase()->Add(encoded_file_descriptor, size): """

    """solution: conda install protobuf=3.8 """

    # ac_tn, ac_tt, ls_tn, ls_tt = lr_curve_extractor('/Users/zeliangsu/Desktop/event')
    gradient_extractor('/media/tomoserver/DATA3/zeliang/github/LRCS-Xlearn/logs/2020_7_15_RESUME_stp_59807_mdl_LRCS11_bs2_ps512_cs3_nc32_do0.0_act_leaky_aug_True_BN_True_mode_classification_lFn_DSC_lrtyperamp_decay0.0001_k0.3_p1.0_cmt_Na7b_pos3/hour14_gpu1/')
    print('ha')
