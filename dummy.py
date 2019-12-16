import numpy as np
import pandas as pd
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt

input_path = './dummy/acc_step9999.csv'
output_path = './dummy/acc_step9999_interp_log.csv'

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

def clean(array):
    assert isinstance(array, np.ndarray)
    array[np.where(array == np.nan)] = 1e-9
    array[np.where(array == 0)] = 1e-9
    array[np.where(array == np.inf)] = 1e9
    array[np.where(array == -np.inf)] = -1e9
    return array

lss = np.asarray(pd.read_csv(input_path))
print('Loss/Acc range: {} - {}'.format(np.min(lss), np.max(lss)))
print('Out range: {} - {}'.format(np.min(np.log(lss)), np.max(np.log(lss))))
lss = np.log(lss)
x_mesh = np.linspace(-1, 1, 51)
y_mesh = np.linspace(-1, 1, 51)
xx, yy = np.meshgrid(x_mesh, y_mesh)

csv_interp(xx, yy, lss, output_path)

##################
#
# plot
#
##################
lss_interp = np.asarray(pd.read_csv(output_path))
x_mesh = np.linspace(-1, 1, 51 * 5)
y_mesh = np.linspace(-1, 1, 51 * 5)
xx, yy = np.meshgrid(x_mesh, y_mesh)

fig, ax = plt.subplots(1)
cs = ax.contour(xx, yy, lss_interp[:, -1].reshape(255, 255))
plt.clabel(cs, inline=1, fontsize=10)
plt.show()






