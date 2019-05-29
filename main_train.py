import tensorflow as tf
import numpy as np
import datetime
import os

from input import inputpipeline
from model import model, model_lite
from train import train

# params
hyperparams = {
    'patch_size': 72,
    'batch_size': 3200,  # ps40:>1500 GPU allocation warning ps96:>200 GPU allocation warning
    'nb_epoch': 10,
    'nb_batch': None,
    'conv_size': 5,
    'nb_conv': 48,
    'learning_rate': 0.000001,  #should use smaller learning rate when decrease batch size
    'dropout': 0.5,
    'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
    'hour': '{}'.format(datetime.datetime.now().hour),
    'device_option': 'specific_gpu:0'
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
nodes = model_lite(train_inputs,
                   test_inputs,
                   hyperparams['patch_size'],
                   hyperparams['batch_size'],
                   hyperparams['conv_size'],
                   hyperparams['nb_conv'],
                   learning_rate=hyperparams['learning_rate'],
                   )


# print number of params
print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))


if not os.path.exists('./logs/{}/hour{}/'.format(hyperparams['date'], hyperparams['hour'])):
    try:
        os.mkdir('./logs/{}/hour{}/'.format(hyperparams['date'], hyperparams['hour']))
    except:
        if not os.path.exists('./logs/{}/'.format(hyperparams['date'])):
            os.mkdir('./logs/{}/'.format(hyperparams['date']))
        os.mkdir('./logs/{}/hour{}/'.format(hyperparams['date'], hyperparams['hour']))

hyperparams['nb_batch'] = len(hyperparams['totrain_files']) // hyperparams['batch_size']

# start training
train(nodes, train_inputs, test_inputs, hyperparams, device_option=hyperparams['device_option'], save_step=20)



