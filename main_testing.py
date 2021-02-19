from evaluate import testing_recursive, testing
import argparse
import pandas as pd
import numpy as np
from PIL import Image

# force this to run on main
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tstd', '--test_dir', type=str, metavar='', default='./test/', required=True,
                        help='where to find the testing dataset')
    parser.add_argument('-ckpt', '--ckpt_path', type=str, metavar='', required=True, help='.meta path')
    parser.add_argument('-sd', '--save_dir', type=str, metavar='', required=True, help='indicate the save path')
    args = parser.parse_args()
    d = args.test_dir

    save_pb_dir = '/'.join(d.split('/')[:-2]) + '/pb/'

    paths = {
        'ckpt_path': args.ckpt_path,
        'test_dir': d,
        'save_pb_dir': save_pb_dir,
        'working_dir': '/'.join(d.split('/')[:-1]) + '/',
    }

    hyperparams = {
        'mode': 'classification',
        'feature_map': False,
        'loss_option': 'Dice',
        'batch_normalization': True,
        'device_option': 'cpu',
        'norm': 1e3,
    }

    acc, lss, y, label = testing(paths=paths,
                          hyper=hyperparams,
                          numpy=True,
                          )
    pd.DataFrame({'acc': [acc], 'lss': [lss]}).to_csv(args.save_dir+'test_score.csv')
    y = np.squeeze(y)
    label = np.squeeze(label)

    assert y.shape == label.shape
    if len(y.shape) == 2:
        Image.fromarray(y).save(args.save_dir+'0_pred.tif')
        Image.fromarray(label).save(args.save_dir+'0_label.tif')
    elif len(y.shape) == 3:
        for i in range(y.shape[0]):
            Image.fromarray(y[i]).save(args.save_dir+'{}_pred.tif'.format(i))
            Image.fromarray(label[i]).save(args.save_dir+'{}_label.tif'.format(i))