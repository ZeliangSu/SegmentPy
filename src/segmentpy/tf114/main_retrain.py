import datetime
import argparse

from train import main_train
from util import exponential_decay, ramp_decay, check_N_mkdir
from input import coords_gen, get_max_nb_cls
from hypParser import string_to_hypers

import numpy as np
import re
import os
import shutil
import json

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
    parser.add_argument('-dv', '--device', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-cmt', '--comment', type=str, metavar='', required=False, help='extra comment')
    parser.add_argument('-st', '--save_model_step', type=int, default=500, required=False,
                        help='save the model every X step')
    parser.add_argument('-tb', '--save_tb', type=int, default=50, required=False,
                        help='save the histograms of gradients and weights for the training every X step')

    # misc
    parser.add_argument('-trnd', '--train_dir', type=str, metavar='', default='./train/', required=False, help='where to find the training dataset')
    parser.add_argument('-vald', '--val_dir', type=str, metavar='', default='./valid/', required=False, help='where to find the valid dataset')
    parser.add_argument('-tstd', '--test_dir', type=str, metavar='', default='./test/', required=False, help='where to find the testing dataset')
    parser.add_argument('-logd', '--log_dir', type=str, metavar='', default='./logs/', required=False,
                        help='where to find the testing dataset')

    # other changeable params
    parser.add_argument('-lr', '--lr_decay_type', type=str, metavar='', required=False,
                        help='learning rate schedule e.g. ramp, exp, const')
    parser.add_argument('-ilr', '--init_lr', type=float, metavar='', required=False,
                        help='starting learning rate e.g. 0.001, 1e-4')
    parser.add_argument('-klr', '--lr_decay_ratio', type=float, metavar='', required=False,
                        help='the decay ratio e.g. 0.1')
    parser.add_argument('-plr', '--lr_period', type=float, metavar='', required=False, help='decay every X epoch')
    parser.add_argument('-nodes', '--restore_nodes', type=str, default='', nargs='+', required=False, help='restrict nodes to restore')
    parser.add_argument('-corr', '--correction', type=float, metavar='', default=1e3, required=False, help='img * correction')
    parser.add_argument('-stch', '--stretch', type=float, metavar='', default=2.0, required=False, help='parameter for stretching')
    parser.add_argument('-stride', '--sampling_stride', type=int, metavar='', default=5, required=False,
                        help='indicate the step/stride with which we sample')
    parser.add_argument('-cond', '--condition', type=float, metavar='', default=0.001, required=False,
                        help='parameter for stretching')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(-1 if args.device == 'cpu' else args.device)
    # params
    hyperparams = {
        ############### model ###################
        'from_ckpt': args.from_checkpoint,
        'model': str(re.search('\_mdl\_([a-zA-Z]+\d*)\_', args.from_checkpoint).group(1)),
        'dropout': float(re.search('\_do(\d+)', args.from_checkpoint).group(1)),
        'augmentation': bool(re.search('\_aug\_(True|False)\_', args.from_checkpoint).group(1)),
        'batch_normalization': bool(re.search('\_BN\_(True|False)\_', args.from_checkpoint).group(1)),
        'activation': str(re.search('act\_(relu|leaky|sigmoid|tanh)', args.from_checkpoint).group(1)),
        'loss_option': str(re.search('lossFn\_(DSC|cross\_entropy|MSE)', args.from_checkpoint).group(1)),
        'mode': str(re.search('\_mode\_(classification|regression)\_', args.from_checkpoint).group(1)),

        ############### hyper-params ##############
        'patch_size': int(re.search('\_ps(\d+)', args.from_checkpoint).group(1)),
        'batch_size': int(re.search('\_bs(\d+)', args.from_checkpoint).group(1)),
        'conv_size': int(re.search('\_cs(\d+)', args.from_checkpoint).group(1)),
        'nb_conv': int(re.search('\_nc(\d+)', args.from_checkpoint).group(1)),
        'old_init_lr': float(re.search('decay({})\\_'.format(ultimate_numeric_pattern), args.from_checkpoint).group(1)),
        'init_lr': float(re.search('decay({})\\_'.format(ultimate_numeric_pattern), args.from_checkpoint).group(1)) if args.init_lr is None else args.init_lr,
        'old_lr_decay_ratio': float(re.search('k({})\\_'.format(ultimate_numeric_pattern), args.from_checkpoint).group(1)),
        'lr_decay_ratio': float(re.search('k({})\\_'.format(ultimate_numeric_pattern), args.from_checkpoint).group(1)) if args.lr_decay_ratio is None else args.lr_decay_ratio,
        'old_lr_period': float(re.search('p({})\\_'.format(ultimate_numeric_pattern), args.from_checkpoint).group(1)),
        'lr_period': float(re.search('p({})\\_'.format(ultimate_numeric_pattern), args.from_checkpoint).group(1)) if args.lr_period is None else args.lr_period,

        ############### misc #####################
        'device_option': args.device,
        'nb_epoch': args.nb_epoch,
        'save_step': 500 if args.save_model_step is None else args.save_model_step,
        'save_summary_step': 50 if args.save_tb is None else args.save_tb,
        'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
        'hour': '{}'.format(datetime.datetime.now().hour),

        'train_dir': args.train_dir,
        'val_dir': args.val_dir,
        'test_dir': args.test_dir,

        'restore_nodes': args.restore_nodes,
        'correction': args.correction,
        'stretch': args.stretch,
        'condition': args.condition,
    }

    logger.info(hyperparams)
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
                                             valid_dir=hyperparams['val_dir'],
                                             window_size=hyperparams['patch_size'],
                                             train_test_ratio=0.9,
                                             stride=args.sampling_stride,
                                             nb_batch=None,
                                             batch_size=hyperparams['batch_size'])

    # calculate nb_batch
    hyperparams['nb_batch'] = hyperparams['input_coords'].get_nb_batch()

    # get learning rate schedule
    if hyperparams['lr_decay_type'] == 'exp':
        hyperparams['learning_rate'] = exponential_decay(
            hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
            hyperparams['init_lr'],
            k=hyperparams['lr_decay_ratio'],
        )  # float32 or np.array of programmed learning rate
    elif hyperparams['lr_decay_type'] == 'ramp':
        hyperparams['learning_rate'] = ramp_decay(
            hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
            hyperparams['nb_batch'],
            hyperparams['init_lr'],
            k=hyperparams['lr_decay_ratio'],
            period=hyperparams['lr_period'],
        )  # float32 or np.array of programmed learning rate
    elif hyperparams['lr_decay_type'] == 'const':
        hyperparams['learning_rate'] = np.zeros(hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1)) + hyperparams['init_lr']
    else:
        raise NotImplementedError('Not implemented learning rate schedule: {}'.format(hyperparams['lr_decay_type']))

    # coordinations gen
    # hyperparams['input_coords'] = coords_gen(train_dir=hyperparams['train_dir'],
    #                                          valid_dir=hyperparams['val_dir'],
    #                                          window_size=hyperparams['patch_size'],
    #                                          train_test_ratio=0.9,
    #                                          stride=args.sampling_stride,
    #                                          nb_batch=None,
    #                                          batch_size=hyperparams['batch_size'])

    # calculate nb_batch
    # hyperparams['nb_batch'] = hyperparams['input_coords'].get_nb_batch()

    # name the log directory
    hyperparams['folder_name'] = \
        '{}{}_RESUME_stp_{}_mdl_{}_bs{}_ps{}_cs{}_nc{}_do{}_act_{}_aug_{}_BN_{}_mode_{}_lFn_{}_lrtype{}_i{}_k{}_p{}_newi_{}_newk_{}_newp_{}_cmt_{}/hour{}_{}/'.format(
            args.log_dir,
            hyperparams['date'],
            string_to_hypers(hyperparams['from_ckpt']).get_step(),
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
            hyperparams['old_init_lr'],
            hyperparams['old_lr_decay_ratio'],
            hyperparams['old_lr_period'],
            hyperparams['init_lr'],
            hyperparams['lr_decay_ratio'],
            hyperparams['lr_period'],
            args.comment.replace(' ', '_'),
            hyperparams['hour'],
            'gpu{}'.format(args.device) if args.device!='cpu' else 'cpu'
        )

    hyperparams['new_params'] = {
        'mdl': hyperparams['model'],
        'bs': hyperparams['batch_size'],
        'ws': hyperparams['patch_size'],
        'cs': hyperparams['conv_size'],
        'nc': hyperparams['nb_conv'],
        'do': hyperparams['dropout'],
        'act': hyperparams['activation'],
        'aug': hyperparams['augmentation'],
        'bn': hyperparams['batch_normalization'],
        'mode': hyperparams['mode'],
        'lFn': hyperparams['loss_option'],
        'ldtype': hyperparams['lr_decay_type'],
        'ldinit': hyperparams['init_lr'],
        'ldratio': hyperparams['lr_decay_ratio'],
        'ldperiod': hyperparams['lr_period'],
        'cmt': args.comment.replace(' ', '_'),
    }
    print(hyperparams['new_params'])

    check_N_mkdir(os.path.join(hyperparams['folder_name']))
    with open(os.path.join(hyperparams['folder_name'], 'from_ckpt.txt'), 'w') as f:
        f.write(hyperparams['from_ckpt'])
    with open(os.path.join(hyperparams['folder_name'], 'new_params.json'), 'w') as f:
        json.dump(hyperparams['new_params'], f)


    check_N_mkdir(hyperparams['folder_name'] + 'copy/')
    shutil.copytree(hyperparams['train_dir'], hyperparams['folder_name'] + 'copy/train/')
    shutil.copytree(hyperparams['val_dir'], hyperparams['folder_name'] + 'copy/val/')
    shutil.copytree(hyperparams['test_dir'], hyperparams['folder_name'] + 'copy/test/')

    hyperparams['max_nb_cls'] = get_max_nb_cls(hyperparams['train_dir'])[1]
    main_train(hyperparams, grad_view=True, nb_classes=hyperparams['max_nb_cls'], resume=args.restore_nodes)

