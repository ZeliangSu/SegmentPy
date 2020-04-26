import datetime
import os
import argparse

from input import inputpipeline_V2
from model import *
from train import train_test
from util import check_N_mkdir, exponential_decay, ramp_decay
from input import coords_gen

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.WARNING)

# argparser
parser = argparse.ArgumentParser('launch main.py')
parser.add_argument('-nc', '--nb_conv', type=int, metavar='', required=True, help='minimum number of convolution per layer e.g. 16, 32, 48')
parser.add_argument('-bs', '--batch_size', type=int, metavar='', required=True, help='number of images per batch e.g. 8, 200, 300 (impact the model size)')
parser.add_argument('-ws', '--window_size', type=int, metavar='', required=True, help='size of the scanning window e.g. 128, 256, 512')
parser.add_argument('-ep', '--nb_epoch', type=int, metavar='', required=True, help='number of epoch')
parser.add_argument('-cs', '--conv_size', type=int, metavar='', required=True, help='kernel size e.g. 3x3, 5x5')
parser.add_argument('-lr', '--learning_rate', type=str, metavar='', required=True, help='learning rate schedule e.g. ramp, exp, const')
parser.add_argument('-ilr', '--init_lr', type=float, metavar='', required=True, help='starting learning rate e.g. 0.001, 1e-4')
parser.add_argument('-klr', '--lr_decay_param', type=float, metavar='', required=True, help='the decay ratio e.g. 0.1')
parser.add_argument('-plr', '--lr_period', type=float, metavar='', required=True, help='decay every X epoch')
parser.add_argument('-bn', '--batch_norm', type=bool, metavar='', required=True, help='use batch normalization or not')
parser.add_argument('-do', '--dropout_prob', type=float, metavar='', required=True, help='dropout probability for the Dense-NN part')
parser.add_argument('-ag', '--augmentation', type=bool, metavar='', required=True, help='use augmentation on the input pipeline')
parser.add_argument('-fn', '--loss_fn', type=str, metavar='', required=True, help='indicate the loss function e.g. DSC, CE')
parser.add_argument('-af', '--activation_fn', type=str, metavar='', required=True, help='activation function e.g. relu, leaky')
parser.add_argument('-mdl', '--model', type=str, metavar='', required=True, help='which model to use e.g. LRCS, Xlearn, Unet')
parser.add_argument('-mode', '--mode', type=str, metavar='', required=True, help='regression/classification')
parser.add_argument('-dv', '--device', type=int, metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
parser.add_argument('-st', '--save_model_step', type=int, metavar='', required=False, help='save the model every X step')
parser.add_argument('-tb', '--save_tb', type=int, metavar='', required=False, help='save the histograms of gradients and weights for the training every X step')
args = parser.parse_args()
print(args)


if __name__ == '__main__':
    tf.reset_default_graph()
    # params
    hyperparams = {
        'patch_size': args.window_size,
        'batch_size': args.batch_size,  #Xlearn < 20, Unet < 20 saturate GPU memory
        'nb_epoch': args.nb_epoch,
        'nb_batch': None,
        'conv_size': args.conv_size,
        'nb_conv': args.nb_conv,
        'dropout': args.dropout_prob,
        'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
        'hour': '{}'.format(datetime.datetime.now().hour),
        'device': args.device,
        'augmentation': args.augmentation,
        'activation': args.activation_fn,
        'batch_normalization': args.batch_norm,
        'save_step': 500 if args.save_model_step is not None else args.save_model_step,
        'save_summary_step': 50 if args.save_tb is not None else args.save_tb,
        'folder_name': None,
        'model': args.model,
        'mode': args.mode,
        'loss_option': args.loss_fn,
    }

    hyperparams['input_coords'] = coords_gen(train_dir='./raw/',
                                             test_dir='./testdata/',
                                             window_size=hyperparams['patch_size'],
                                             train_test_ratio=0.9,
                                             stride=5,
                                             nb_batch=None,
                                             batch_size=hyperparams['batch_size'])

    # calculate nb_batch
    hyperparams['nb_batch'] = hyperparams['input_coords'].get_nb_batch()

    # get learning rate schedule
    if args.learning_rate == 'exp':
        hyperparams['learning_rate'] = exponential_decay(
            hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
            args.init_lr,
            k=args.lr_decay_param
        )  # float32 or np.array of programmed learning rate

    elif args.learning_rate == 'ramp':
        hyperparams['learning_rate'] = ramp_decay(
            hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
            hyperparams['nb_batch'],
            args.init_lr,
            k=args.lr_decay_param,
            period=args.lr_period,
        )  # float32 or np.array of programmed learning rate

    elif args.learning_rate == 'const':
        hyperparams['learning_rate'] = np.zeros(hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1)) + args.init_lr
    else:
        raise NotImplementedError('Not implemented learning rate schedule: {}'.format(args.learning_rate))

    # name the log directory
    hyperparams['folder_name'] = \
        './logs/{}_bs{}_ps{}_lr{}_cs{}_nc{}_do{}_act_{}_aug_{}_BN_{}_mdl_{}_mode_{}_lossFn_{}_{}decay{}_k{}_p{}_comment{}/hour{}_gpu{}/'.format(
        hyperparams['date'],
        hyperparams['batch_size'],
        hyperparams['patch_size'],
        hyperparams['learning_rate'] if not isinstance(hyperparams['learning_rate'], np.ndarray) else 'programmed',
        hyperparams['conv_size'],
        hyperparams['nb_conv'],
        hyperparams['dropout'],
        hyperparams['activation'],
        str(hyperparams['augmentation']),
        str(hyperparams['batch_normalization']),
        hyperparams['model'],
        hyperparams['mode'],
        args.loss_fn, args.learning_rate,
        args.init_lr, args.lr_decay_param,
        args.lr_period,
        '_fix_coord_gen',  #note: here put your special comment
        hyperparams['hour'],
        hyperparams['device']
    )

    # init input pipeline
    train_inputs = inputpipeline_V2(hyperparams['batch_size'], suffix='train', augmentation=hyperparams['augmentation'], mode='classification')
    test_inputs = inputpipeline_V2(hyperparams['batch_size'], suffix='test', mode='classification')

    # define other placeholder
    if hyperparams['dropout'] is not None:
        drop_prob = tf.placeholder(tf.float32, name='dropout_prob')
    else:
        drop_prob = tf.placeholder_with_default(1.0, [], name='dropout_prob')

    if hyperparams['batch_normalization']:
        BN_phase = tf.placeholder_with_default(False, (), name='BN_phase')
    else:
        BN_phase = False

    # init model
    lr = tf.placeholder(tf.float32, name='learning_rate')
    list_placeholders = [drop_prob, lr, BN_phase]
    train_nodes = classification_nodes(pipeline=train_inputs,
                                       placeholders=list_placeholders,
                                       model_name=hyperparams['model'],
                                       patch_size=hyperparams['patch_size'],
                                       batch_size=hyperparams['batch_size'],
                                       conv_size=hyperparams['conv_size'],
                                       nb_conv=hyperparams['nb_conv'],
                                       activation=hyperparams['activation'],
                                       batch_norm=hyperparams['batch_normalization'],
                                       loss_option=hyperparams['loss_option'],
                                       is_training=True,
                                       device=hyperparams['device']
                                       )
    # fixme: the following load 2 modes in one gpu
    test_nodes = classification_nodes(pipeline=test_inputs,
                                      placeholders=list_placeholders,
                                      model_name=hyperparams['model'],
                                      patch_size=hyperparams['patch_size'],
                                      batch_size=hyperparams['batch_size'],
                                      conv_size=hyperparams['conv_size'],
                                      nb_conv=hyperparams['nb_conv'],
                                      activation=hyperparams['activation'],
                                      batch_norm=hyperparams['batch_normalization'],
                                      loss_option=hyperparams['loss_option'],
                                      is_training=False,
                                      device=hyperparams['device']
                                      )

    # print number of params
    print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

    # create logs folder
    check_N_mkdir('./logs/')

    # start training
    train_test(train_nodes, test_nodes, train_inputs, test_inputs, hyperparams)



