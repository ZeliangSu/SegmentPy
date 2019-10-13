import tensorflow as tf
import numpy as np
import datetime
import os

from input import inputpipeline
from model import *
from train import train
from util import check_N_mkdir

# params
hyperparams = {
    'patch_size': 72,
    'batch_size': 300,  #Xlearn < 20, Unet < 20 saturate GPU memory
    'nb_epoch': 100,
    'nb_batch': None,
    'conv_size': 5,
    'nb_conv': 80,
    'learning_rate': 1e-4,  #should use smaller learning rate when decrease batch size
    'dropout': 0.1,
    'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
    'hour': '{}'.format(datetime.datetime.now().hour),
    'device_option': 'specific_gpu:1',
    'augmentation': True,
    'activation': 'leaky',
    'save_step': 1000,
}


# get list of file names
hyperparams['totrain_files'] = [os.path.join('./proc/train/{}/'.format(hyperparams['patch_size']),
                              f) for f in os.listdir('./proc/train/{}/'.format(hyperparams['patch_size'])) if f.endswith('.h5')]
hyperparams['totest_files'] = [os.path.join('./proc/test/{}/'.format(hyperparams['patch_size']),
                             f) for f in os.listdir('./proc/test/{}/'.format(hyperparams['patch_size'])) if f.endswith('.h5')]

# init input pipeline
train_inputs = inputpipeline(hyperparams['batch_size'], suffix='train', augmentation=hyperparams['augmentation'])
test_inputs = inputpipeline(hyperparams['batch_size'], suffix='test')

# init model
nodes = model_xlearn(train_inputs,
                   test_inputs,
                   hyperparams['patch_size'],
                   hyperparams['batch_size'],
                   hyperparams['conv_size'],
                   hyperparams['nb_conv'],
                   learning_rate=hyperparams['learning_rate'],
                   activation=hyperparams['activation'],
                   )


# print number of params
print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

# create logs folder
check_N_mkdir('./logs/')

# calculate nb_batch
hyperparams['nb_batch'] = len(hyperparams['totrain_files']) // hyperparams['batch_size']

# start training
train(nodes, train_inputs, test_inputs, hyperparams, device_option=hyperparams['device_option'], save_step=hyperparams['save_step'])



