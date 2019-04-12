import tensorflow as tf
import numpy as np
import datetime
import os

from input import inputpipeline
from model import model
from train import train

# params
hyperparams = {
    'patch_size': 72,
    'batch_size': 100,  # ps40:>1500 GPU allocation warning ps96:>200 GPU allocation warning
    'nb_epoch': 20,
    'nb_batch': None,
    'conv_size': 3,
    'nb_conv': 32,
    'learning_rate': 0.000001,  #should use smaller learning rate when decrease batch size
    'dropout': 0.5,
    'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
    'hour': '{}'.format(datetime.datetime.now().hour),
    'device_option': 'specific_gpu:1'
}


# get list of file names
hyperparams['totrain_files'] = [os.path.join('./proc/train/{}/'.format(hyperparams['patch_size']),
                              f) for f in os.listdir('./proc/train/{}/'.format(hyperparams['patch_size'])) if f.endswith('.h5')]
hyperparams['totest_files'] = [os.path.join('./proc/test/{}/'.format(hyperparams['patch_size']),
                             f) for f in os.listdir('./proc/test/{}/'.format(hyperparams['patch_size'])) if f.endswith('.h5')]

# init input pipeline
train_inputs = inputpipeline(hyperparams['batch_size'], suffix='train')
test_inputs = inputpipeline(hyperparams['batch_size'], suffix='test')

# init model
nodes = model(train_inputs,
              test_inputs,
              hyperparams['patch_size'],
              hyperparams['batch_size'],
              hyperparams['conv_size'],
              hyperparams['nb_conv'],
              learning_rate=hyperparams['learning_rate'],
              )

# print number of params
print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))


if not os.path.exists('./logs/{}/'.format(hyperparams['date'])):
    os.mkdir('./logs/{}/'.format(hyperparams['date']))
    
if not os.path.exists('./logs/{}/hour{}/'.format(hyperparams['date'], hyperparams['hour'])):
    os.mkdir('./logs/{}/hour{}/'.format(hyperparams['date'], hyperparams['hour']))
hyperparams['nb_batch'] = len(hyperparams['totrain_files']) // hyperparams['batch_size']

# start training
train(nodes, train_inputs, test_inputs, hyperparams)



