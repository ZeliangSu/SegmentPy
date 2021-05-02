import os
import subprocess
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse

parser = argparse.ArgumentParser('launch main.py')
parser.add_argument('-ckpt', '--ckpt_path', type=str, metavar='', required=True, help='.meta path')
parser.add_argument('-raw', '--raw_dir', type=str, metavar='', required=True, help='raw tomograms folder path')
parser.add_argument('-pred', '--pred_dir', type=str, metavar='', required=True, help='where to put the segmentation')
parser.add_argument('-corr', '--correction', type=float, metavar='', required=True, help='manually correct the input image by a coefficient')
args = parser.parse_args()

if __name__ == '__main__':
    # os.system("mpiexec -n 3 python inference.py")

    ##################### below cannot capture the callback
    # note: 'mpiexec' for MPICH and 'mpirun' for mpi4py
    # note: Popen is more powerful handling spawn process then os.sys
    #  https://stackoverflow.com/a/33784179/9217178
    subprocess.Popen([
        'mpirun', '--use-hwthread-cpus', 'python', os.path.join(os.path.dirname(__file__), 'inference.py'),
        '-ckpt', args.ckpt_path,
        '-raw', args.raw_dir,
        '-pred', args.pred_dir,

    ])
    ##################### above cannot capture the callback

    # os.system("mpirun -n 79 python inference.py -ckpt {} -raw {} -pred {} -corr {}".format(  #mpirun --use-hwthread-cpus python inference.py -ckpt {} -raw {} -pred {}
    #     args.ckpt_path,
    #     args.raw_dir,
    #     args.pred_dir,
    #     args.correction,
    # ))

