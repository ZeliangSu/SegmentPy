import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse

parser = argparse.ArgumentParser('launch main.py')
parser.add_argument('-ckpt', '--ckpt_path', type=str, metavar='', required=True, help='.meta path')
parser.add_argument('-raw', '--raw_path', type=str, metavar='', required=True, help='raw tomograms folder path')
parser.add_argument('-pred', '--pred_path', type=str, metavar='', required=True, help='where to put the segmentation')
args = parser.parse_args()

if __name__ == '__main__':
    # os.system("mpiexec -n 3 python inference.py")
    os.system("mpiexec --use-hwthread-cpus python inference.py -ckpt {} -raw {} -pred {}".format(
        args.ckpt_path,
        args.raw_path,
        args.pred_path
    ))