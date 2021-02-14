from itertools import product
import datetime
from tqdm import tqdm
import subprocess
import argparse
import os
import pandas as pd
import numpy as np
from skopt import gp_minimize, load
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram
from skopt import callbacks
from skopt.callbacks import CheckpointSaver
from skopt.plots import plot_convergence


# logging
import logging
import log

logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)


def get_test_acc(log_dir: str):
    assert os.path.isdir(log_dir)
    acc = None
    try:
        if not os.path.exists(log_dir+'curves/'):
            acc = 0
            # todo: this way of finding score is not accurate
            logger.warning('[SegmentPy]: could not find the test score folder')
        else:
            acc = pd.read_csv(log_dir+'curves/test_score.csv', header=True).acc.mean()

    except Exception as e:
        logger.error(e)

    return acc


class hyp(object):
    def __init__(self, hyp):
        self.lb_initial_lr = hyp['lbilr']
        self.ub_initial_lr = hyp['ubilr']
        self.lb_lr_decay = hyp['lbdecay']
        self.ub_lr_decay = hyp['ubdecay']
        self.list_kernel_size = hyp['lks']
        self.lb_batch_size = hyp['lbbs']
        self.ub_batch_size = hyp['ubbs']
        self.lb_nb_conv = hyp['lbnbconv']
        self.ub_nb_conv = hyp['ubnbconv']


class bayesianOpt:
    def __init__(self, hyps: hyp):
        logger.debug(hyps)
        self.ilr = Real(low=hyps.lb_initial_lr, high=hyps.ub_initial_lr, prior='log-uniform', name='init_learning_rate')
        self.lr_decay = Real(low=hyps.lb_lr_decay, high=hyps.ub_lr_decay, prior='uniform', name='lr_decay_ratio')
        self.ks = Categorical(hyps.list_kernel_size, name='kernel_size')
        self.batch_size = Integer(low=hyps.lb_batch_size, high=hyps.ub_batch_size, name='batch_size')
        self.nb_conv = Integer(low=hyps.lb_nb_conv, high=hyps.ub_nb_conv, name='nb_conv')

        self.dimensions = [
            self.lr_decay, self.batch_size, self.nb_conv, self.ilr, self.ks
        ]

        self.default = [
            0.3, 2, 32, 1e-4, 3
        ]

    def sim(self, *args, **kwargs):
        logger.debug(args)
        logger.debug(kwargs)
        return float(np.random.random(1))

    def Baye_search(self, func, space: list):
        checkpoint_saver = CheckpointSaver(args.hyper_ckpt)
        rlt = gp_minimize(func,
                          dimensions=space,
                          acq_func='EI',
                          n_calls=50,
                          n_random_starts=3,
                          # callback=[checkpoint_saver],
                          random_state=42)
        logger.debug(rlt)
        logger.debug(rlt)
        plot_convergence(rlt)
        return rlt

    def Baye_search_resume(self, func, path: str, space):
        assert os.path.exists(path)
        ckpt = load(path)
        checkpoint_saver = CheckpointSaver(args.hyper_ckpt)
        rlt = gp_minimize(func,
                          dimensions=space,
                          x0=ckpt.x_iters,
                          y0=ckpt.func_vals,
                          n_calls=20,
                          n_random_starts=3,
                          # callback=[checkpoint_saver],
                          random_state=42)
        logger.debug(rlt)
        plot_convergence(rlt)
        return rlt

    def blackBox_func(self, hyperparams: dict):
        # todo: hyperparams should change to a object
        # launch main train
        logger.debug(hyperparams)
        self.date = '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month,
                                 datetime.datetime.now().day),
        self.hour = '{}'.format(datetime.datetime.now().hour),
        try:
            terminal = [
                'python', 'main_train.py',
                '-nc', hyperparams['nb_conv'],
                '-bs', hyperparams['batch_size'],
                '-ws', 512,
                '-ep', 5,
                '-cs', hyperparams['conv_size'],
                '-lr', 'ramp',
                '-ilr', hyperparams['initial_lr'],
                '-klr', hyperparams['lr_decay_ratio'],
                '-plr', 1,
                '-bn', 'True',
                '-do', 0.0,
                '-ag', 'True',
                '-fn', 'DSC',
                '-af', 'leaky',
                '-mdl', 'LRCS11',
                '-mode', 'classification',
                '-dv', args.device,
                '-st', 500,
                '-tb', 50,
                '-cmt', args.comment,
                '-trnd', args.train_dir,
                '-vald', args.val_dir,
            ]
            process = subprocess.Popen(
                terminal,
            )
            o, e = process.communicate()
            if e:
                logger.error('[segmentpy]: catched exception\n')
                logger.error(e)

        except Exception as e:
            logger.error('[segmentpy]: catched exception\n')
            logger.error(e)

        # get test score
        log_dir = '{}/{}_mdl_{}_bs{}_ps{}_cs{}_nc{}_do{}_act_{}_aug_{}_BN_{}_mode_{}_lossFn_{}_lrtype{}_decay{}_k{}_p{}_comment_{}/hour{}_gpu{}/'.format(
            args.log_dir,
            self.date,
            hyperparams['model'],
            hyperparams['batch_size'],
            hyperparams['patch_size'],
            hyperparams['conv_size'],
            hyperparams['nb_conv'],
            hyperparams['dropout'],
            hyperparams['activation'],
            str(hyperparams['augmentation']),
            str(hyperparams['batch_normalization']),
            hyperparams['mode'],
            hyperparams['loss_option'],
            hyperparams['lr_decay_type'],
            hyperparams['initial_lr'],
            hyperparams['lr_decay_ratio'],
            hyperparams['lr_decay_period'],
            args.comment,
            self.hour,
            hyperparams['device']
        )  # todo: reorganize to elsewhere
        acc = get_test_acc(log_dir)

        # return score
        return float(1 - acc)

    def launch(self):
        @use_named_args(dimensions=self.dimensions)
        def wrapper(*args, **kwargs):
            # return self.sim(*args, **kwargs)
            return self.blackBox_func(*args, **kwargs)
        return self.Baye_search(wrapper, self.dimensions)


def grid_search_no_repeat(space: dict):
    # product
    # for h in space:
    # for lr, bs, dr, ks, nc in tqdm(product()):
    #     pass
    # return combo to the interface
    pass



if __name__ == '__main__':
    # copy this
    # python hyper.py -dv 0 -srchTyp baye -hypCkpt './test.pkl' -lbilr 1e-5 -ubilr 1e-3 -lbdecay 0.1 -ubdecay 0.5 -lks 3 5 -lbbs 2 -ubbs 12 -lbnbconv 16 -ubnbconv 48 -trnd './train/' -vald './val/' -tstd './test/', -cmt test

    parser = argparse.ArgumentParser()
    parser.add_argument('-dv', '--device', type=int, required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-srchTyp', '--search_type', type=str, required=True, help='baye|grid')
    parser.add_argument('-hypCkpt', '--hyper_ckpt', type=str, required=True, help='where to save')

    #############################
    parser.add_argument('-lbilr', '--lb_initial_lr', type=float, required=True, help='lower_bound_initial_lr')
    parser.add_argument('-ubilr', '--ub_initial_lr', type=float, required=True, help='upper_bound_initial_lr')
    parser.add_argument('-lbdecay', '--lb_lr_decay', type=float, required=True, help='lower_bound_lr_decay')
    parser.add_argument('-ubdecay', '--ub_lr_decay', type=float, required=True, help='upper_bound_lr_decay')
    parser.add_argument('-lks', '--list_kernel_size', nargs='+', required=True, help='list_bound_kernel_size')
    parser.add_argument('-lbbs', '--lb_batch_size', type=int, required=True, help='lower_bound_batch_size')
    parser.add_argument('-ubbs', '--ub_batch_size', type=int, required=True, help='upper_bound_batch_size')
    parser.add_argument('-lbnbconv', '--lb_nb_conv', type=int, required=True, help='lower_bound_nb_conv')
    parser.add_argument('-ubnbconv', '--ub_nb_conv', type=int, required=True, help='upper_bound_nb_conv')

    #############################
    parser.add_argument('-trnd', '--train_dir', type=str, required=False, default='./train/', help='where to get the training data')
    parser.add_argument('-vald', '--val_dir', type=str, required=False, default='./valid/', help='where to get the valid data')
    parser.add_argument('-tstd', '--test_dir', type=str, required=False, default='./test/', help='where to get the testing data')
    parser.add_argument('-logd', '--log_dir', type=str, required=False, default='./logs', help='where to save models')
    parser.add_argument('-cmt', '--comment', type=str, required=False, help='comment')

    args = parser.parse_args()

    hyp_dict = {
        'lbilr': args.lb_initial_lr,
        'ubilr': args.ub_initial_lr,
        'lbdecay': args.lb_lr_decay,
        'ubdecay': args.ub_lr_decay,
        'lks': args.list_kernel_size,
        'lbbs': args.lb_batch_size,
        'ubbs': args.ub_batch_size,
        'lbnbconv': args.lb_nb_conv,
        'ubnbconv': args.ub_nb_conv,
    }

    hyp1 = hyp(hyp_dict)

    baye1 = bayesianOpt(hyp1)
    baye1.launch()




