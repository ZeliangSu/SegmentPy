import tensorflow as tf
import numpy as np
import datetime
import os

from input import inputpipeline
from model import *
from train import train_test
from util import check_N_mkdir

import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)

# params
hyperparams = {
    'patch_size': 512,
    'batch_size': 8,  #Xlearn < 20, Unet < 20 saturate GPU memory
    'nb_epoch': 100,
    'nb_batch': None,
    'conv_size': 3,
    'nb_conv': 48,
    'learning_rate': 1e-4,  #float or np.array of programmed learning rate
    'dropout': 0.1,
    'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
    'hour': '{}'.format(datetime.datetime.now().hour),
    'device_option': 'specific_gpu:0',
    'second_device': 'specific_gpu:1',
    'augmentation': True,
    'activation': 'leaky',
    'save_step': 500,
    'save_summary_step': 50,
    'folder_name': None,
    'model': 'LRCS',
    'mode': 'classification',
}

hyperparams['folder_name'] = './logs/{}_bs{}_ps{}_lr{}_cs{}_nc{}_do{}_act_{}_aug_{}_mdl_{}_mode_{}_comment_{}/hour{}/'.format(
    hyperparams['date'],
    hyperparams['batch_size'],
    hyperparams['patch_size'],
    hyperparams['learning_rate'] if not isinstance(hyperparams['learning_rate'], np.ndarray) else 'programmed',
    hyperparams['conv_size'],
    hyperparams['nb_conv'],
    hyperparams['dropout'],
    hyperparams['activation'],
    str(hyperparams['augmentation']),
    hyperparams['model'],
    'Add_softmax_then_DSC',  #note: here put your special comment
    hyperparams['hour'],
)

# get list of file names
hyperparams['totrain_files'] = [os.path.join('./proc/train/{}/'.format(hyperparams['patch_size']),
                              f) for f in os.listdir('./proc/train/{}/'.format(hyperparams['patch_size'])) if f.endswith('.h5')]
hyperparams['totest_files'] = [os.path.join('./proc/test/{}/'.format(hyperparams['patch_size']),
                             f) for f in os.listdir('./proc/test/{}/'.format(hyperparams['patch_size'])) if f.endswith('.h5')]

# init input pipeline
train_inputs = inputpipeline(hyperparams['batch_size'], suffix='train', augmentation=hyperparams['augmentation'], mode='classification')
test_inputs = inputpipeline(hyperparams['batch_size'], suffix='test', mode='classification')
drop_prob = tf.placeholder(tf.float32, name='dropout_prob')
lr = tf.placeholder(tf.float32, name='learning_rate')
BN_phase = tf.placeholder_with_default(False, (), name='BN_phase')
list_placeholders = [drop_prob, lr, BN_phase]
# init model
train_nodes = classification_nodes(pipeline=train_inputs,
                    placeholders=list_placeholders,
                    model_name=hyperparams['model'],
                    patch_size=hyperparams['patch_size'],
                    batch_size=hyperparams['batch_size'],
                    conv_size=hyperparams['conv_size'],
                    nb_conv=hyperparams['nb_conv'],
                    activation=hyperparams['activation'],
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
                   is_training=False,
                   )


# print number of params
print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

# create logs folder
check_N_mkdir('./logs/')

# calculate nb_batch
hyperparams['nb_batch'] = len(hyperparams['totrain_files']) // hyperparams['batch_size']

# start training
train_test(train_nodes, test_nodes, train_inputs, test_inputs, hyperparams)



