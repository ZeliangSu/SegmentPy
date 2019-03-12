import numpy as np
import h5py
import random

class MBGDHelper:
    '''Mini Batch Grandient Descen helper'''
    def __init__(self, batch_size, patch_size):
        self.i = 0
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.epoch_len = self._epoch_len()
        self.order = np.arange(self.epoch_len) #data has been pre-shuffle
        self.onoff = 0
    def next_batch(self):
        try:
            try:
                with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
                    X = f['X'][self.order[self.i * self.batch_size: (self.i + 1) * self.batch_size, ]].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
                    y = f['y'][self.order[self.i * self.batch_size: (self.i + 1) * self.batch_size, ]].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
                self.i += 1
                return X, y
            except:
                print('\n***Load last batch')
                with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
                    modulo = f['X'].shape % self.batch_size
                    X = f['X'][self.order[-modulo:, ]].reshape(modulo, self.patch_size, self.patch_size, 1)
                    y = f['y'][self.order[-modulo:, ]].reshape(modulo, self.patch_size, self.patch_size, 1)
                self.i += 1
                return X, y
        except:
            print('\n***epoch finished')
            self.onoff = 1
            pass

    def _epoch_len(self):
        with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
            print('Total epoch number is {}'.format(f['X'].shape[0]))
            return f['X'].shape[0]

    def get_epoch(self):
        return self.epoch_len

    def shuffle(self):
        np.random.shuffle(self.order)
        print('shuffled datas')