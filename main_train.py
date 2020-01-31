import datetime
import os

from input import inputpipeline
from model import *
from train import train_test
from util import check_N_mkdir, exponential_decay, ramp_decay

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)

l_nc = [16, 48]
l_cs = [3]
l_lr = ['ramp']  #1e-5, 'ramp', 'exp'
init_lr = [1e-4]; k = [0.1]; period = [1]   #exp: k=1e-5 strong decay after 4 epoch ramp: 0.5
l_BN = [True]
l_do = [0.1]

for _do in l_do:
    for _BN in l_BN:
        for n, _lr in enumerate(l_lr):
            for _cs in l_cs:
                for _nc in l_nc:
                    tf.reset_default_graph()
                    # params
                    hyperparams = {
                        'patch_size': 512,
                        'batch_size': 8,  #Xlearn < 20, Unet < 20 saturate GPU memory
                        'nb_epoch': 5,
                        'nb_batch': None,
                        'conv_size': _cs,
                        'nb_conv': _nc,
                        'dropout': _do,
                        'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
                        'hour': '{}'.format(datetime.datetime.now().hour),
                        'device_option': 'specific_gpu:0',
                        'second_device': 'specific_gpu:1',
                        'augmentation': True,
                        'activation': 'leaky',
                        'batch_normalization': _BN,
                        'save_step': 500,
                        'save_summary_step': 50,
                        'folder_name': None,
                        'model': 'LRCS',
                        'mode': 'classification',
                        'loss_option': 'DSC',
                    }

                    # get list of file names
                    hyperparams['totrain_files'] = [os.path.join('./proc/train/{}/'.format(hyperparams['patch_size']),
                                                                 f) for f in
                                                    os.listdir('./proc/train/{}/'.format(hyperparams['patch_size'])) if
                                                    f.endswith('.h5')]
                    hyperparams['totest_files'] = [os.path.join('./proc/test/{}/'.format(hyperparams['patch_size']),
                                                                f) for f in
                                                   os.listdir('./proc/test/{}/'.format(hyperparams['patch_size'])) if
                                                   f.endswith('.h5')]

                    # calculate nb_batch
                    hyperparams['nb_batch'] = len(hyperparams['totrain_files']) // hyperparams['batch_size']

                    # get learning rate schedule
                    if isinstance(_lr, str):
                        if _lr == 'exp':
                            hyperparams['learning_rate'] = exponential_decay(
                                hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
                                init_lr[n],
                                k=k[n]
                            )  # float32 or np.array of programmed learning rate
                        elif _lr == 'ramp':
                            hyperparams['learning_rate'] = ramp_decay(
                                hyperparams['nb_epoch'] * (hyperparams['nb_batch'] + 1),
                                hyperparams['nb_batch'],
                                init_lr[n],
                                k=k[n],
                                period=period[n],
                            )  # float32 or np.array of programmed learning rate
                        else:
                            raise NotImplementedError('Not implemented learning rate schedule: {}'.format(_lr))
                    else:
                        hyperparams['learning_rate'] = _lr

                    # name the log directory
                    hyperparams['folder_name'] = './logs/{}_bs{}_ps{}_lr{}_cs{}_nc{}_do{}_act_{}_aug_{}_BN_{}_mdl_{}_mode_{}_comment_{}/hour{}/'.format(
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
                        'DSC_rampdecay{}_k{}_p{}_wrapperWithoutMinmaxscaler_augWith_test_aug_GreyVar'.format(init_lr[n], k[n], period[n]),
                        #note: here put your special comment
                        hyperparams['hour'],
                    )

                    # init input pipeline
                    train_inputs = inputpipeline(hyperparams['batch_size'], suffix='train', augmentation=hyperparams['augmentation'], mode='classification')
                    test_inputs = inputpipeline(hyperparams['batch_size'], suffix='test', mode='classification')

                    # define placeholder
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
                                                       )

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
                                                      )

                    # print number of params
                    print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

                    # create logs folder
                    check_N_mkdir('./logs/')

                    # start training
                    train_test(train_nodes, test_nodes, train_inputs, test_inputs, hyperparams)



