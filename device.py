from tensorflow.python.client import device_lib
import os


def get_available_gpus(l: list):
    if not os.path.exists('./device.txt'):
        local_devices = device_lib.list_local_devices()
        for x in local_devices:
            if x.device_type == 'GPU':
                l.append(int(x.name.split(':')[-1]))
        with open('./device.txt', 'w') as f:
            for dv in l:
                f.write('{}\n'.format(dv))
    return l


if __name__ == '__main__':
    get_available_gpus([])


