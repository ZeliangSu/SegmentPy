import datetime
import numpy as np
import argparse
import os
import shutil
import platform
import subprocess
import json

from train import main_train
from util import exponential_decay, ramp_decay, check_N_mkdir, boolean_string, get_latest_training_number
from input import coords_gen, get_max_nb_cls
from score_extractor import lr_curve_extractor, df_to_csv

# logging
import logging
import log

logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)

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
    parser.add_argument('-trnd', '--train_dir', type=str, metavar='', default=os.path.join(os.path.dirname(__file__), 'train'), required=False,
                        help='where to find the training dataset')
    parser.add_argument('-vald', '--val_dir', type=str, metavar='', default=os.path.join(os.path.dirname(__file__), 'valid'), required=False,
                        help='where to find the valid dataset')
    parser.add_argument('-tstd', '--test_dir', type=str, metavar='', default=os.path.join(os.path.dirname(__file__), 'test'), required=False,
                        help='where to find the testing dataset')
    parser.add_argument('-logd', '--log_dir', type=str, metavar='', default=os.path.join(os.path.dirname(__file__), 'log'), required=False,
                        help='where to find the testing dataset')
    parser.add_argument('-stride', '--sampling_stride', type=int, metavar='', default=5, required=False,
                        help='indicate the step/stride with which we sample')
    parser.add_argument('-corr', '--correction', type=float, metavar='', default=1e3, required=False,
                        help='img * correction')
    parser.add_argument('-stch', '--stretch', type=float, metavar='', default=2.0, required=False,
                        help='parameter for stretching')
    parser.add_argument('-cond', '--condition', type=float, metavar='', default=0.001, required=False,
                        help='parameter for stretching')

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
            'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month,
                                      datetime.datetime.now().day),
            'hour': '{}'.format(datetime.datetime.now().hour),

            'train_dir': args.train_dir,
            'val_dir': args.val_dir,
            'test_dir': args.test_dir,
            'correction': 1e3,
            'stretch': 2.0,
            'condition': args.condition,
        }

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
            hyperparams['learning_rate'] = np.zeros(
                hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1)) + args.init_lr
        else:
            raise NotImplementedError('Not implemented learning rate schedule: {}'.format(args.lr_decay_type))

        # get last training number
        latest_number = get_latest_training_number(args.log_dir) + 1
        # name the log directory
        hyperparams['folder_name'] = os.path.join(
            '{}{}_{}_mdl_{}_bs{}_ps{}_cs{}_nc{}_do{}_act_{}_aug_{}_BN_{}_mode_{}_lossFn_{}_lrtype{}_decay{}_k{}_p{}_comment_{}'.format(
                args.log_dir,
                latest_number,
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
                'hour{}_{}'.format(hyperparams['hour'],
                'gpu{}'.format(args.device) if args.device != 'cpu' else 'cpu')))

        check_N_mkdir(hyperparams['folder_name'])
        with open(os.path.join(hyperparams['folder_name'], 'HPs.json'), 'w') as file:
            json.dump({'corr': hyperparams['correction'],
                       'str': hyperparams['stretch'],
                       'cond': hyperparams['condition'],
                       'gap': args.sampling_stride,
                       }, file)

    except Exception as e:
        logger.warning(
            '\n\n(main_train.py)%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%THERE IS A PARSER ERROR, but still run with default values%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')
        logger.error(e)
    ###################################################################


    # backup dataset
    check_N_mkdir(os.path.join(hyperparams['folder_name'], 'copy'))
    shutil.copytree(hyperparams['train_dir'], os.path.join(hyperparams['folder_name'], 'copy','train'))
    shutil.copytree(hyperparams['val_dir'], os.path.join(hyperparams['folder_name'], 'copy', 'val'))
    shutil.copytree(hyperparams['test_dir'], os.path.join(hyperparams['folder_name'], 'copy', 'test'))

    # try:
    hyperparams['max_nb_cls'] = get_max_nb_cls(hyperparams['train_dir'])[1]
    start_time = datetime.datetime.now()
    main_train(hyperparams, grad_view=True, nb_classes=hyperparams['max_nb_cls'])
    train_time = (datetime.datetime.now() - start_time) / 3600
    # save lr_curves
    check_N_mkdir(hyperparams['folder_name'] + 'curves')
    ac_tn, _, ls_tn, _ = lr_curve_extractor(os.path.join(hyperparams['folder_name'], 'train'))
    _, ac_val, _, ls_val = lr_curve_extractor(os.path.join(hyperparams['folder_name'], 'test'))
    best_step = ac_val.step.loc[ac_val.value.argmax()]
    # best_step=0
    df_to_csv(os.path.join(hyperparams['folder_name'], 'curves'), ac_tn, ac_val, ls_tn, ls_val)
    with open(os.path.join(hyperparams['folder_name'], 'curves', 'train_time.csv'), 'w') as f:
        f.write('{} hours'.format(train_time.seconds/3600))

    # testing
    logger.debug(best_step)
    logger.debug(hyperparams['folder_name'])
    logger.debug(args.test_dir)
    p = subprocess.Popen(['python', 'main_testing.py',
                          '-tstd', args.test_dir,
                          '-ckpt', os.path.join(hyperparams['folder_name'], '/curves/best_model'),
                          '-sd', os.path.join(hyperparams['folder_name'], 'test_score.csv')])
    o, e = p.communicate()

