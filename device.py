from _taskManager.opening_logic import opening_logic

from PyQt5.QtWidgets import QApplication
from tensorflow.python.client import device_lib
from tqdm import tqdm
from time import sleep
import sys


def get_available_gpus():
    l = []
    window = QApplication(sys.argv)

    ui = opening_logic()
    # ui.show()
    # window.exec_()
    # ui.log.setText('Scanning available devices...')

    #
    local_devices = device_lib.list_local_devices()
    for dv in local_devices:
        if dv.device_type == 'CPU' or dv.device_type == "XLA_GPU":
            ui.log.setText('Cannot find available GPUs, use CPU instead...')
        elif dv.device_type == 'GPU':
            ui.log.setText('Found GPU: {}'.format(dv.name))

    # write devices
    for x in tqdm(local_devices):
        if x.device_type == 'GPU' or x.device_type == "XLA_GPU":
            l.append(int(x.name.split(':')[-1]))

    with open('./device.txt', 'w') as f:
        for dv in l:
            f.write('{}\n'.format(dv))

    # quit
    sys.exit(0)


if __name__ == '__main__':
    get_available_gpus()


