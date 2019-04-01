from proc import preprocess, preprocess_V2
from train import test_train
import tensorflow as tf
import h5py
import os
import multiprocessing as mp


preproc = {
    'indir': './raw',
    'stride': 3,
    'patch_size': 96,  # should be multiple of 8
    # 'batch_size': 1000,
    'mode': 'h5',
    'shuffle': True,
    'traintest_split_rate': 0.95
}

# preprocess(**preproc)
preprocess_V2(**preproc)
