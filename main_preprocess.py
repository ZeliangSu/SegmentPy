from proc import preprocess, preprocess_V2
from train import test_train
import tensorflow as tf
import h5py
import os
import multiprocessing as mp
import matplotlib.pyplot as plt

preproc = {
    'indir': './raw',
    'stride': 15,
    'patch_size': 40,  # should be multiple of 8
    # 'batch_size': 1000,
    'mode': 'h5',
    'shuffle': True
}

# preprocess(**preproc)
preprocess_V2(**preproc)
