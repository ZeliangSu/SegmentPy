from proc import preprocess
import tensorflow as tf
import h5py
import os
import multiprocessing as mp


preproc = {
    'indir': './raw',
    'stride': 20,
    'patch_size': 40,  # should be multiple of 8
    # 'batch_size': 1000,
    'mode': 'h5',
    'shuffle': True,
    'traintest_split_rate': 0.95
}

# preprocess(**preproc)
preprocess(**preproc)
