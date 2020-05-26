import datetime
import argparse

from train import main_train
from util import exponential_decay, ramp_decay
from input import coords_gen

import numpy as np
import re
import os

import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)

# for re that detect -1e-5, 0.0001, all type
ultimate_numeric_pattern = '[-+]?(?:(?:\\d*\\.\\d+)|(?:\\d+\\.?))(?:[Ee][+-]?\\d+)?'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--from_checkpoint', type=str, metavar='', required=True, help='full/absolute path to the ckpt is required')
    parser.add_argument('-ep', '--nb_epoch', type=int, metavar='', required=True, help='encore howmany epochs')
    parser.add_argument('-dv', '--device', type=int, metavar='', required=True, help='on which gpu')
    parser.add_argument('-cmt', '--comment', type=str, metavar='', required=False, help='extra comment')
    args = parser.parse_args()

    # params
    hyperparams = {
        ############### model ###################
        'from_ckpt': args.from_checkpoint,
        'model': str(re.search('\_mdl\_([a-zA-Z]+\d+)\_', args.from_checkpoint).group(1)),
        'dropout': float(re.search('\_do(\d+)', args.from_checkpoint).group(1)),
        'augmentation': bool(re.search('\_aug\_(True|False)\_', args.from_checkpoint).group(1)),
        'batch_normalization': bool(re.search('\_BN\_(True|False)\_', args.from_checkpoint).group(1)),
        'activation': str(re.search('\_act\_(\w+)\_', args.from_checkpoint).group(1)),
        'loss_option': str(re.search('\_lossFn\_(\w+)\_', args.from_checkpoint).group(1)),

        ############### hyper-paras ##############
        'patch_size': int(re.search('\_ps(\d+)', args.from_checkpoint).group(1)),
        'batch_size': int(re.search('\_bs(\d+)', args.from_checkpoint).group(1)),
        'conv_size': int(re.search('\_cs(\d+)', args.from_checkpoint).group(1)),
        'nb_conv': int(re.search('\_nc(\d+)', args.from_checkpoint).group(1)),
        'init_lr': int(re.search('decay({})\\_'.format(ultimate_numeric_pattern), args.from_checkpoint).group(1)),
        'lr_decay_ratio': int(re.search('k({})\\_'.format(ultimate_numeric_pattern), args.from_checkpoint).group(1)),
        'lr_period': int(re.search('p({})\\_'.format(ultimate_numeric_pattern), args.from_checkpoint).group(1)),

        ############### misc #####################
        'device_option': args.device,
        'nb_epoch': args.nb_epoch,
        'save_step': 500 if args.save_model_step is None else args.save_model_step,
        'save_summary_step': 50 if args.save_tb is None else args.save_tb,
        'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
        'hour': '{}'.format(datetime.datetime.now().hour),

        'train_dir': './train/',
        'val_dir': './valid/',
        'test_dir': './test/',
    }

    logger.warn('new folder name format will be change and adapted in the next version')
    try:
        # old folder name format
        hyperparams['lr_decay_type'] = str(re.search('\_([a-z]+)decay', args.from_checkpoint).group(1))
    except Exception as e:
        logger.debug(e)
        # new format
        hyperparams['lr_decay_type'] = str(re.search('\_lrtype([a-z]+)\_', args.from_checkpoint).group(1))

    # coordinations gen
    hyperparams['input_coords'] = coords_gen(train_dir=hyperparams['train_dir'],
                                             test_dir=hyperparams['test_dir'],
                                             window_size=hyperparams['patch_size'],
                                             train_test_ratio=0.9,
                                             stride=5,
                                             nb_batch=None,
                                             batch_size=hyperparams['batch_size'])

    # calculate nb_batch
    hyperparams['nb_batch'] = hyperparams['input_coords'].get_nb_batch()

    # get learning rate schedule
    if hyperparams['lr_decay_type'] == 'exp':
        hyperparams['learning_rate'] = exponential_decay(
            hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
            args.init_lr,
            k=args.lr_decay_param
        )  # float32 or np.array of programmed learning rate
    elif hyperparams['lr_decay_type'] == 'ramp':
        hyperparams['learning_rate'] = ramp_decay(
            hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
            hyperparams['nb_batch'],
            args.init_lr,
            k=args.lr_decay_param,
            period=args.lr_period,
        )  # float32 or np.array of programmed learning rate
    elif hyperparams['lr_decay_type'] == 'const':
        hyperparams['learning_rate'] = np.zeros(hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1)) + args.init_lr
    else:
        raise NotImplementedError('Not implemented learning rate schedule: {}'.format(args.learning_rate))

    # coordinations gen
    hyperparams['input_coords'] = coords_gen(train_dir=hyperparams['train_dir'],
                                             test_dir=hyperparams['test_dir'],
                                             window_size=hyperparams['patch_size'],
                                             train_test_ratio=0.9,
                                             stride=5,
                                             nb_batch=None,
                                             batch_size=hyperparams['batch_size'])

    # calculate nb_batch
    hyperparams['nb_batch'] = hyperparams['input_coords'].get_nb_batch()

    # name the log directory
    hyperparams['folder_name'] = \
        './logs/{}_RESUME_mdl_{}_bs{}_ps{}_cs{}_nc{}_do{}_act_{}_aug_{}_BN_{}_mode_{}_lossFn_{}_lrtype{}_decay{}_k{}_p{}_comment_{}/hour{}_gpu{}/'.format(
            hyperparams['date'],
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
            hyperparams['init_lr'],
            hyperparams['lr_decay_ratio'],
            hyperparams['lr_period'],
            args.comment.replace(' ', '_'),
            hyperparams['hour'],
            hyperparams['device']
        )

    with open(os.path.join(hyperparams['folder_name'], 'from_ckpt.txt')) as f:
        f.write(hyperparams['from_ckpt'])

    main_train(hyperparams, resume=True)

