from proc import preprocess, preprocess_V2
import tensorflow as tf
import h5py
import os
import multiprocessing as mp
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)

preproc = {
    'indir': './raw/',
    'stride': 100,
    'patch_size': 512,  # should be multiple of 8
    'shuffle': True,
    'traintest_split_rate': 0.9
}

preprocess_V2(**preproc)
