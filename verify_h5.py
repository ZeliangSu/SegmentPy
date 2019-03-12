import matplotlib.pyplot as plt
import h5py

with h5py.File('./proc/40.h5', 'r') as f:
    print(f['X'].shape)
    for i in range(10):
        plt.figure(i)
        plt.imshow(f['X'][i*500, ])
        plt.figure(i + 6)
        plt.imshow(f['y'][i*500, ])
        plt.show()
