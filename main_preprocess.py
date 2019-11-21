from proc import preprocess
import tensorflow as tf
import h5py
import os
import multiprocessing as mp


preproc = {
    'indir': './raw',
    'stride': 5,
    'patch_size': 512,  # should be multiple of 8
    'mode': 'h5',
    'shuffle': True,
    'traintest_split_rate': 0.9
}

preprocess(**preproc)
