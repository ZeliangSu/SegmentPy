from itertools import product
from tqdm import tqdm
import subprocess
import argparse


def grid_search():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dv', '--device', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-vald', '--val_dir', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-trnd', '--train_dir', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')
    parser.add_argument('-cmt', '--comment', metavar='', required=True, help='which GPU to use e.g. -1 use CPU')

    lr = ['1e-4', '1e-5', '1e-6']
    batch_size = ['2', '6']
    decay_ratio = ['0.3', '0.5']
    kernel_size = ['3', '5']
    nb_conv = ['16', '32', '48']
    args = parser.parse_args()
    for lr, bs, dr, ks, nc in tqdm(product(lr, batch_size, decay_ratio, kernel_size, nb_conv)):
        terminal = [
            'python', 'main_train.py',
            '-nc', nc,
            '-bs', bs,
            '-ws', '512',
            '-ep', '5',
            '-cs', ks,
            '-lr', 'ramp',
            '-ilr', lr,
            '-klr', dr,
            '-plr', '1',
            '-bn', 'True',
            '-do', '0.0',
            '-ag', 'True',
            '-fn', 'DSC',
            '-af', 'leaky',
            '-mdl', 'LRCS11',
            '-mode', 'classification',
            '-dv', args.device,
            '-st', '500',
            '-tb', '50',
            '-cmt', args.comment,  #'pristine'
            '-trnd', args.train_dir,  # '/media/tomoserver/DATA3/zeliang/github/LRCS-Xlearn/paper/CNT_recharged/train/'
            '-vald', args.val_dir,  # '/media/tomoserver/DATA3/zeliang/github/LRCS-Xlearn/paper/CNT_recharged/valid/'
        ]
        process = subprocess.Popen(
            terminal,
        )
        stdout, stderr = process.communicate()
    pass