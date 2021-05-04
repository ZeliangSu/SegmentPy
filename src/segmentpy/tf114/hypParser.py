import re
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import tensorflow as tf
# from tensorflow.core.framework import graph_pb2

from segmentpy.tf114.score_extractor import lr_curve_extractor, gradient_extractor

import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger('root')
logger.setLevel(logging.DEBUG)


ultimate_numeric_pattern = '[-+]?(?:(?:\\d*\\.\\d+)|(?:\\d+\\.?))(?:[Ee][+-]?\\d+)?'


class string_to_hypers:
    # todo: this could be an object
    def __init__(self, folder_name: str):
        self.folder_name = folder_name
        if not self.folder_name.endswith('/'):
            self.folder_name += '/'
        self.param_list = {}

    def parse(self):
        _param_list = {}
        _param_list['model'] = self.get_model()
        if _param_list['model'] is None:
            logger.debug(self.folder_name)
            raise ValueError('Failed to parse the folder name')
        _param_list['batch_size'] = self.get_batch_size()
        _param_list['window_size'] = self.get_window_size()
        _param_list['kernel_size'] = self.get_kernel_size()
        _param_list['nb_conv'] = self.get_nb_conv()
        _param_list['act_fn'] = self.get_act_fn()
        _param_list['BatchNorm'] = self.get_BN()
        _param_list['augmentation'] = self.get_aug()
        _param_list['dropout'] = self.get_dropout()
        _param_list['loss_fn'] = self.get_loss_fn()
        _param_list['lr_decay_type'] = self.get_decay_type()
        _param_list['lr_init'] = self.get_lr_init()
        _param_list['lr_decay'] = self.get_lr_decay_ratio()
        _param_list['lr_period'] = self.get_lr_period()
        _param_list['comment'] = self.get_comment()
        _param_list['mode'] = self.get_cls_or_reg()
        return _param_list

    def get_step(self):
        step = re.search('step(\d+)', self.folder_name)
        if step is not None:
            step = step.group(1)
        return step

    def get_model(self):
        mdl = re.search('mdl\_([A-Za-z]+\d*)', self.folder_name)
        if mdl is not None:
            mdl = mdl.group(1)
        return mdl

    def get_batch_size(self):
        bs = re.search('bs(\d+)', self.folder_name)
        if bs is not None:
            bs = bs.group(1)
        return bs

    def get_window_size(self):
        ws = re.search('ps(\d+)', self.folder_name)
        if ws is not None:
            ws = ws.group(1)
        return ws

    def get_kernel_size(self):
        ks = re.search('cs(\d+)', self.folder_name)
        if ks is not None:
            ks = ks.group(1)
        return ks

    def get_nb_conv(self):
        nc = re.search('nc(\d+)', self.folder_name)
        if nc is not None:
            nc = nc.group(1)
        return nc

    def get_dropout(self):
        do = re.search('do(\d*.\d+)', self.folder_name)
        if do is not None:
            do = do.group(1)
        return do

    def get_act_fn(self):
        af = re.search('act\_(relu|leaky|sigmoid|tanh)', self.folder_name)
        if af is not None:
            af = af.group(1)
        return af

    def get_BN(self):
        bn = re.search('BN\_(True|False)', self.folder_name)
        if bn is not None:
            bn = bn.group(1)
        return bn

    def get_aug(self):
        ag = re.search('aug\_(True|False)', self.folder_name)
        if ag is not None:
            ag = ag.group(1)
        return ag

    def get_loss_fn(self):
        lss = re.search('lossFn\_(DSC|cross\_entropy|MSE)', self.folder_name)
        if lss is not None:
            lss = lss.group(1)
        return lss

    def get_decay_type(self):
        dt = re.search('lrtype\_*(exp|ramp|constant)', self.folder_name)
        if dt is not None:
            dt = dt.group(1)
        return dt

    def get_lr_init(self):
        init = re.search('decay({})'.format(ultimate_numeric_pattern), self.folder_name)
        if init is not None:
            init = init.group(1)
        return init

    def get_lr_decay_ratio(self):
        dk = re.search('\_k({})'.format(ultimate_numeric_pattern), self.folder_name)
        if dk is not None:
            dk = dk.group(1)
        return dk

    def get_lr_period(self):
        dp = re.search('\_p({})'.format(ultimate_numeric_pattern), self.folder_name)
        if dp is not None:
            dp = dp.group(1)
        return dp

    def get_comment(self):
        cmt = re.search('comment\_(.+)\/?', self.folder_name)
        if cmt is not None:
            cmt = cmt.group(1)
        return cmt

    def get_cls_or_reg(self):
        cls_reg = re.search('mode\_(classification|regression)', self.folder_name)
        if cls_reg is not None:
            cls_reg = cls_reg.group(1)
        return cls_reg

    def folder_level(self):
        if re.search('hour\_gpu-?\d+\/?$', self.folder_name) is not None:
            # level 2: e.g. .../hour13_gpu1/
            logger.debug('detected a level 2 folder name')
            return 2
        elif re.search('\/.+comment\_\w+\/', self.folder_name) is not None:
            # level 1: e.g. ...comment_None/
            logger.debug('detected a level 1 folder name')
            return 1
        else:
            return 0


class string_to_data(string_to_hypers):
    def __init__(self, folder_name: str):
        super().__init__(folder_name=folder_name)

    def extract_learning_curves(self):
        self.acc_tns = []
        self.acc_tts = []
        self.lss_tns = []
        self.lss_tts = []

        folder_level = self.folder_level()
        if folder_level == 1:
            for tfev_folder in tqdm(os.listdir(self.folder_name)):
                if not tfev_folder.startswith('.'):
                    tfev_folder += '/'
                    acc_tn, _, lss_tn, _ = lr_curve_extractor(os.path.join(self.folder_name, tfev_folder, 'train/'))
                    _, acc_tt, _, lss_tt = lr_curve_extractor(os.path.join(self.folder_name, tfev_folder, 'test/'))

                    self.acc_tns.append(acc_tn)
                    self.acc_tts.append(acc_tt)
                    self.lss_tns.append(lss_tn)
                    self.lss_tts.append(lss_tt)

        elif folder_level == 2:
            acc_tn, _, lss_tn, _ = lr_curve_extractor(os.path.join(self.folder_name, 'train/'))
            _, acc_tt, _, lss_tt = lr_curve_extractor(os.path.join(self.folder_name, 'test/'))

            self.acc_tns.append(acc_tn)
            self.acc_tts.append(acc_tt)
            self.lss_tns.append(lss_tn)
            self.lss_tts.append(lss_tt)
        else:
            logger.info(ValueError('cannot understand the passed path of the TFevents'))
            self.acc_tns.append(None)
            self.acc_tts.append(None)
            self.lss_tns.append(None)
            self.lss_tts.append(None)

    def hyper_to_DataFrame(self):
        if not hasattr(self, 'acc_tns'):
            self.extract_learning_curves()

        # init df
        params = self.parse()
        cols = [k for k in params.keys()]
        cols += ['acc_tns', 'acc_tts', 'lss_tns', 'lss_tts', 'acc_tns_max', 'acc_tts_max', 'lss_tns_min', 'lss_tts_min']
        df = pd.DataFrame(columns=cols)

        # write df
        for i in range(len(self.acc_tns)):
            tmp = pd.DataFrame({
                k: [v] for k, v in params.items()
            })
            tmp['acc_tns'] = [self.acc_tns[i]]
            tmp['acc_tts'] = [self.acc_tts[i]]
            tmp['lss_tns'] = [self.lss_tns[i]]
            tmp['lss_tts'] = [self.lss_tts[i]]

            tmp['acc_tns_max'] = [self.acc_tns[i].value.max() if isinstance(self.acc_tns[i], pd.DataFrame) else None]
            tmp['acc_tts_max'] = [self.acc_tts[i].value.max() if isinstance(self.acc_tts[i], pd.DataFrame) else None]

            tmp['lss_tns_min'] = [self.lss_tns[i].value[1:].min() if isinstance(self.lss_tns[i], pd.DataFrame) else None]
            tmp['lss_tts_min'] = [self.lss_tts[i].value.min() if isinstance(self.lss_tts[i], pd.DataFrame) else None]

            df = df.append(tmp, ignore_index=True)  # note: append is not an in-place operation
        return df


class graph_to_tensor_name:
    def __int__(self, graph):
        assert isinstance(graph, tf.GraphDef) or isinstance(graph, tf.Graph)
        self.graph = graph

    def get_weight(self):
        self.weight_names = []

        if isinstance(self.graph, tf.GraphDef):
            for n in self.graph.node:
                if re.search('\/w$', n) is not None:
                    self.weight_names.append(n)
        elif isinstance(self.graph, tf.Graph):
            for n in self.graph.as_graph_def().node:
                if re.search('\/w$', n) is not None:
                    self.weight_names.append(n)

        else:
            raise ValueError('expecting a tf.graph, tf.graphdef but get other things')

        return self.weight_names

    def get_activations(self):
        self.activation_name = []

        if isinstance(self.graph, tf.GraphDef):
            for n in self.graph.node:
                if re.search('\/(relu|leaky|sigmoid|tanh)', n) is not None:
                    self.activation_name.append(n)

        elif isinstance(self.graph, tf.Graph):
            for n in self.graph.as_graph_def().node:
                if re.search('\/(relu|leaky|sigmoid|tanh)', n) is not None:
                    self.activation_name.append(n)

        return self.activation_name

    def get_output_node_name(self):
        if isinstance(self.graph, tf.GraphDef):
            for n in self.graph.node:
                if re.search('\/identity', n) is not None:
                    self.output_node_name = n
        elif isinstance(self.graph, tf.Graph):
            for n in self.graph.as_graph_def().node:
                if re.search('\/identity', n) is not None:
                    self.output_node_name = n
        else:
            raise ValueError('did not find the output node in the graph')

        return self.output_node_name


if __name__ == '__main__':
    pd_df = pd.DataFrame()
    path = '/media/tomoserver/DATA3/zeliang/github/paper_ML/broader_gSearch_LRCS/'
    for folder in os.listdir(path):
        if not folder.startswith('.'):  # MacOS: avoid './.DS_Store/'
            hypers = string_to_data(path + folder)
            tmp = hypers.hyper_to_DataFrame()
            # pd_df.columns = tmp.columns
            pd_df = pd_df.append(tmp, ignore_index=True)
    pd_df.to_csv(path+'broader_gSearch_LRCS11_summary.csv')
    print('finished')

