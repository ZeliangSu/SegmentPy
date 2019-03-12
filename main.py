from proc import preprocess
from train import test_train
from model import test
import tensorflow as tf
import h5py
import os
import multiprocessing as mp
import matplotlib.pyplot as plt

preproc = {
    'dir': './raw',
    'stride': 10,
    'patch_size': 128, # multiple of 8
    'batch_size': 1000,
    'mode': 'h5',
    'shuffle': True
}

for i in range(5):
    with h5py.File('./proc/40.h5', 'r') as f:
        plt.figure(i)
        plt.imshow(f['X'][i, ])
        plt.figure(i+5)
        plt.imshow(f['y'][i, ])
        plt.show()
