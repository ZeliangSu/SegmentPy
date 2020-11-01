import datetime
import numpy as np
import argparse
import os
import shutil
import platform

from train import main_train
from util import exponential_decay, ramp_decay, check_N_mkdir, boolean_string
from input import coords_gen, get_max_nb_cls

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.WARNING)

if platform.system() == 'Darwin':
    # mpi problem: https://stackoverflow.com/questions/55714135/how-to-properly-fix-the-following-openmp-error
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# force to run this on main
if __name__ == '__main__':
    # argparser
    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--nb_conv', type=int, metavar='', required=True,
                        help='minimum number of convolution per layer e.g. 16, 32, 48')
    parser.add_argument('-bs', '--batch_size', type=int, metavar='', required=True,
                        help='number of images per batch e.g. 8, 200, 300 (impact the model size)')
    parser.add_argument('-ws', '--window_size', type=int, metavar='', required=True,
                        help='size of the scanning window e.g. 128, 256, 512')
    parser.add_argument('-ep', '--nb_epoch', type=int, metavar='', required=True, help='number of epoch')
    parser.add_argument('-cs', '--conv_size', type=int, metavar='', required=True, help='kernel size e.g. 3x3, 5x5')
    parser.add_argument('-lr', '--lr_decay_type', type=str, metavar='', required=True,
                        help='learning rate schedule e.g. ramp, exp, const')
    parser.add_argument('-ilr', '--init_lr', type=float, metavar='', required=True,
                        help='starting learning rate e.g. 0.001, 1e-4')
    parser.add_argument('-klr', '--lr_decay_ratio', type=float, metavar='', required=True,
                        help='the decay ratio e.g. 0.1')
    parser.add_argument('-plr', '--lr_period', type=float, metavar='', required=True, help='decay every X epoch')
    parser.add_argument('-bn', '--batch_norm', type=str, metavar='', required=True,
                        help='use batch normalization or not')
    parser.add_argument('-do', '--dropout_prob', type=float, metavar='', required=True,
                        help='dropout probability for the Dense-NN part')
    parser.add_argument('-ag', '--augmentation', type=str, metavar='', required=True,
                        help='use augmentation on the input pipeline')
    parser.add_argument('-fn', '--loss_fn', type=str, metavar='', required=True,
                        help='indicate the loss function e.g. DSC, CE')
    parser.add_argument('-af', '--activation_fn', type=str, metavar='', required=True,
                        help='activation function e.g. relu, leaky')
    parser.add_argument('-mdl', '--model', type=str, metavar='', required=True,
                        help='which model to use e.g. LRCS, Xlearn, Unet')
    parser.add_argument('-mode', '--mode', type=str, metavar='', required=True, help='regression/classification')
    parser.add_argument('-dv', '--device', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-st', '--save_model_step', type=int, metavar='', required=False,
                        help='save the model every X step')
    parser.add_argument('-tb', '--save_tb', type=int, metavar='', required=False,
                        help='save the histograms of gradients and weights for the training every X step')
    parser.add_argument('-cmt', '--comment', type=str, metavar='', required=False, help='extra comment')
    parser.add_argument('-trnd', '--train_dir', type=str, metavar='', default='./train/', required=False, help='where to find the training dataset')
    parser.add_argument('-vald', '--val_dir', type=str, metavar='', default='./valid/', required=False, help='where to find the valid dataset')
    parser.add_argument('-tstd', '--test_dir', type=str, metavar='', default='./test/', required=False, help='where to find the testing dataset')
    parser.add_argument('-stride', '--sampling_stride', type=int, metavar='', default=5, required=False, help='indicate the step/stride with which we sample')
    parser.add_argument('-corr', '--correction', type=float, metavar='', default=1e3, required=False, help='img * correction')
    parser.add_argument('-stch', '--stretch', type=float, metavar='', default=2.0, required=False, help='parameter for stretching')

    try:
        args = parser.parse_args()
        print(args)
        print('aug:', boolean_string(args.augmentation))
        print('BN:', boolean_string(args.batch_norm))
        # note: copy this in terminal for debugging
        #  'python main_train.py -mdl LRCS -nc 32 -bs 8 -ws 512 -ep 5 -cs 3 -lr ramp -ilr 1e-4 -klr 0.3 -plr 1 -bn True -do 0.1 -ag True -fn leaky -af DSC -mode classification -dv 0 -st 500 -tb 50 -cmt None

        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(-1 if args.device == 'cpu' else args.device)

        # params
        hyperparams = {
            ############### model ###################
            'model': args.model,
            'mode': args.mode,
            'dropout': args.dropout_prob,
            'augmentation': boolean_string(args.augmentation),
            'batch_normalization': boolean_string(args.batch_norm),
            'activation': args.activation_fn,
            'loss_option': args.loss_fn,

            ############### hyper-paras ##############
            'patch_size': args.window_size,
            'batch_size': args.batch_size,
            'conv_size': args.conv_size,
            'nb_conv': args.nb_conv,

            ############### misc #####################
            'nb_epoch': args.nb_epoch,
            'device': 'cpu' if args.device == 'cpu' else args.device,
            'save_step': 500 if args.save_model_step is None else args.save_model_step,
            'save_summary_step': 50 if args.save_tb is None else args.save_tb,
            'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
            'hour': '{}'.format(datetime.datetime.now().hour),

            'train_dir': args.train_dir,
            'val_dir': args.val_dir,
            'test_dir': args.test_dir,
            'correction': 1e3,
            'stretch': 2.0
            }

        # coordinations gen
        hyperparams['input_coords'] = coords_gen(train_dir=hyperparams['train_dir'],
                                                 valid_dir=hyperparams['val_dir'],
                                                 window_size=hyperparams['patch_size'],
                                                 train_test_ratio=0.9,
                                                 stride=5,
                                                 nb_batch=None,
                                                 batch_size=hyperparams['batch_size'])

        # calculate nb_batch
        hyperparams['nb_batch'] = hyperparams['input_coords'].get_nb_batch()

        # get learning rate schedule
        if args.lr_decay_type == 'exp':
            hyperparams['learning_rate'] = exponential_decay(
                hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
                args.init_lr,
                k=args.lr_decay_ratio
            )  # float32 or np.array of programmed learning rate
        elif args.lr_decay_type == 'ramp':
            hyperparams['learning_rate'] = ramp_decay(
                hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
                hyperparams['nb_batch'],
                args.init_lr,
                k=args.lr_decay_ratio,
                period=args.lr_period,
            )  # float32 or np.array of programmed learning rate
        elif args.lr_decay_type == 'constant':
            hyperparams['learning_rate'] = np.zeros(hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1)) + args.init_lr
        else:
            raise NotImplementedError('Not implemented learning rate schedule: {}'.format(args.lr_decay_type))

        # name the log directory
        hyperparams['folder_name'] = \
            './logs/{}_mdl_{}_bs{}_ps{}_cs{}_nc{}_do{}_act_{}_aug_{}_BN_{}_mode_{}_lossFn_{}_lrtype{}_decay{}_k{}_p{}_comment_{}/hour{}_{}/'.format(
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
                args.loss_fn,
                args.lr_decay_type,
                args.init_lr,
                args.lr_decay_ratio,
                args.lr_period,
                args.comment.replace(' ', '_'),
                hyperparams['hour'],
                'gpu{}'.format(args.device) if args.device != 'cpu' else 'cpu'
            )

    except Exception as e:
        logger.warning('\n\n(main_train.py)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%THERE IS A PARSER ERROR, but still run with default values%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
        logger.error(e)
        hyperparams = {
            ############### model ###################
            'model': 'Unet4',
            'mode': 'classification',
            'dropout': '0.0',
            'augmentation': True,
            'batch_normalization': 'True',
            'activation': 'leaky',
            'loss_option': 'DSC',

            ############### hyper-paras ##############
            'patch_size': 512,
            'batch_size': 8,
            'conv_size': 3,
            'nb_conv': 32,

            ############### misc #####################
            'nb_epoch': 5,
            'device': 0,  #cpu: -1
            'save_step': 500,
            'save_summary_step': 50,
            'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month,
                                      datetime.datetime.now().day),
            'hour': '{}'.format(datetime.datetime.now().hour),

            'train_dir': args.train_dir if args.train_dir is not None else './train/',
            'val_dir': args.val_dir if args.val_dir is not None else'./valid/',
            'test_dir': args.test_dir if args.test_dir is not None else'./test/',
            'correction': args.correction,
            'stretch': args.stretch
        }

        # coordinations gen
        hyperparams['input_coords'] = coords_gen(train_dir=hyperparams['train_dir'],
                                                 valid_dir=hyperparams['val_dir'],
                                                 window_size=hyperparams['patch_size'],
                                                 train_test_ratio=0.9,
                                                 stride=5,
                                                 nb_batch=None,
                                                 batch_size=hyperparams['batch_size'])

        # calculate nb_batch
        hyperparams['nb_batch'] = hyperparams['input_coords'].get_nb_batch()
        ############### generated #################
        hyperparams['learning_rate'] = ramp_decay(
            hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
            hyperparams['nb_batch'],
            1e-5,
            k=0.3,
            period=1,
        )
        hyperparams['folder_name'] = \
            './logs/DEBUG{}_mdl_{}_bs{}_ps{}_cs{}_nc{}_do{}_act_{}_aug_{}_BN_{}_mode_{}_lossFn_{}_lrtype{}_decay{}_k{}_p{}_comment_{}/hour{}_gpu{}/'.format(
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
                'ramp',
                1e-4,
                0.3,
                1,
                'DEBUG',
                hyperparams['hour'],
                hyperparams['device']
            )

    # backup dataset
    check_N_mkdir(hyperparams['folder_name'] + 'copy/')
    shutil.copytree(hyperparams['train_dir'], hyperparams['folder_name'] + 'copy/train/')
    shutil.copytree(hyperparams['val_dir'], hyperparams['folder_name'] + 'copy/val/')
    shutil.copytree(hyperparams['test_dir'], hyperparams['folder_name'] + 'copy/test/')

    # try:
    hyperparams['max_nb_cls'] = get_max_nb_cls(hyperparams['train_dir'])[1]
    main_train(hyperparams, grad_view=True, nb_classes=hyperparams['max_nb_cls'])

    # except Exception as e:
    #     logger.error(e)
    #     with open(hyperparams['folder_name'] + 'exit_log.txt', 'w') as f:
    #         f.write(str(e))
