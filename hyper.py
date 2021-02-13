from itertools import product
import datetime
from tqdm import tqdm
import subprocess
import argparse
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import gp_minimize, load
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
from skopt.plots import plot_objective, plot_histogram
from skopt import callbacks
from skopt.callbacks import CheckpointSaver


# logging
import logging
import log

logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.WARNING)


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


def grid_search_no_repeat(space: dict):
    # product
    # for h in space:
    # for lr, bs, dr, ks, nc in tqdm(product()):
    #     pass
    # return combo to the interface
    pass


def grid_search(space: list):
    pass


def random_search(space: list):
    pass


def Baye_search(space: list):
    checkpoint_saver = CheckpointSaver(args.hyper_ckpt)
    gp_minimize(blackBox_func,
                dimensions=space,
                acq_func='EI',
                n_calls=3,
                n_random_starts=3,
                callback=[checkpoint_saver],
                random_state=42)


def Baye_search_resume(path: str, space):
    assert os.path.exists(path)
    ckpt = load(path)
    checkpoint_saver = CheckpointSaver(args.hyper_ckpt)
    gp_minimize(blackBox_func,
                dimensions=space,
                x0=ckpt.x_iters,
                y0=ckpt.func_vals,
                n_calls=3,
                n_random_starts=3,
                callback=[checkpoint_saver],
                random_state=42)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dv', '--device', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-srchTyp', '--search_type', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-hypCkpt', '--hyper_ckpt', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    #############################


    #############################
    parser.add_argument('-trnd', '--train_dir', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-vald', '--val_dir', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-tstd', '--test_dir', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-logd', '--log_dir', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-cmt', '--comment', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')

    args = parser.parse_args()

    bs = getattr(args, 'batch_size')

    learning_rate_decay = Real(low=0.1, high=0.5, prior='uniform', name='lr_decay_ratio')
    batch_size = Integer(low=2, high=12, name='batch_size')
    nb_conv = Integer(low=16, high=48, name='nb_conv')
    ilr = Real(low=1e-5, high=1e-3, prior='log-uniform', name='init_learning_rate')
    ks = Categorical([3, 5])

    dimensions = [
        bs,
    ]

    @use_named_args(dimensions=dimensions)
    def blackBox_func(hyperparams: dict):
        # todo: hyperparams should change to a object
        # launch main train
        logger.debug(hyperparams)
        date = '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month,
                                 datetime.datetime.now().day),
        hour = '{}'.format(datetime.datetime.now().hour),
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
                '-plr', hyperparams['lr_decay_period'],
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
            date,
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
            hour,
            hyperparams['device']
        )  # todo: reorganize to elsewhere
        acc = get_test_acc(log_dir)

        # return score
        return 1 - acc
