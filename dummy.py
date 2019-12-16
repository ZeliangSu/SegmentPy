import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
from util import clean

if_log = False
if_plot = True
if_clean_data = False
l_steps = ['0000', 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
l_input_path = ['./dummy/lss_step{}.csv'.format(step) for step in l_steps]
l_output_path = ['./dummy/lss_step{}_interp{}.csv'.format(step, '_log' if if_log else None) for step in l_steps]

##################
#
#  conversion
#
##################


def csv_interp(x_mesh, y_mesh, metrics_tensor, out_path, interp_scope=5):
    new_xmesh = np.linspace(np.min(x_mesh), np.max(x_mesh), interp_scope * x_mesh.shape[0])
    new_ymesh = np.linspace(np.min(y_mesh), np.max(y_mesh), interp_scope * x_mesh.shape[1])
    newxx, newyy = np.meshgrid(new_xmesh, new_ymesh)

    # interpolation
    interpolation = interp2d(x_mesh, y_mesh, metrics_tensor, kind='linear')  #note: cubic doesn't work
    zval = interpolation(new_xmesh, new_ymesh)
    pd.DataFrame({'xcoord': newxx.ravel(),
                  'ycoord': newyy.ravel(),
                  'zval': zval.ravel()}
                 ).to_csv(out_path, index=False, header=True)


for input_path, output_path in zip(l_input_path, l_output_path):
    lss = np.asarray(pd.read_csv(input_path))
    print('Loss/Acc range: {} - {}'.format(np.min(lss), np.max(lss)))
    print('Out range: {} - {}'.format(np.min(np.log(lss)), np.max(np.log(lss))))
    if if_log:
        lss = np.log(lss)
    if if_clean_data:
        lss = clean(lss)
    x_mesh = np.linspace(-1, 1, 51)
    y_mesh = np.linspace(-1, 1, 51)
    xx, yy = np.meshgrid(x_mesh, y_mesh)

    csv_interp(xx, yy, lss, output_path)

    ##################
    #
    # plot
    #
    ##################
    if if_plot:
        lss_interp = np.asarray(pd.read_csv(output_path))
        x_mesh = np.linspace(-1, 1, 51 * 5)
        y_mesh = np.linspace(-1, 1, 51 * 5)
        xx, yy = np.meshgrid(x_mesh, y_mesh)

        fig, ax = plt.subplots(1)
        cs = ax.contour(xx, yy, lss_interp[:, -1].reshape(255, 255))
        plt.clabel(cs, inline=1, fontsize=10)
        plt.savefig(output_path.replace('.csv', '.png'))
        plt.show()






