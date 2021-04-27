import subprocess
import argparse
import re

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)  #changeHere: debug level

from analytic import partialRlt_and_diff, visualize_weights, weights_euclidean_distance,\
    weights_angularity, weights_hists_2excel, tsne_on_weights, tsne_on_bias

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint_path', nargs='+', type=str, required=True, help='.meta path')
    parser.add_argument('-step', '--steps', type=int, nargs='+', required=True, help='which step to retrieve analytics')
    parser.add_argument('-dir', '--data_dir', type=str, required=False, default='./test/', help='which data to use for input')
    parser.add_argument('-type', '--retrieve_type', nargs='+', type=str, default='activation', required=False,
                        help='choose activation/weight/tsne/l2/ang/hist or a combo of them doing -type activation weight euclidean...')
    parser.add_argument('-node', '--conserve_nodes', type=str, nargs='+', required=True, help='enter operation/tensor name to check, can be plural')
    parser.add_argument('-stride', '--stride', type=int, default=50, required=False, help='choose the stride of sampling')
    args = parser.parse_args()
    print(args)

    for ckpt_path in args.checkpoint_path:
        print(ckpt_path)
        model = re.search('mdl_([A-Za-z]*\d*)', ckpt_path).group(1)
        print('model: ', model)
        hyperparams = {
            'model': model,
            'window_size': int(re.search('ps(\d+)', ckpt_path).group(1)),
            'batch_size': int(re.search('bs(\d+)', ckpt_path).group(1)),
            'stride': args.stride,
            'device_option': 'cpu',
            'mode': 'classification',  # todo:
            'batch_normalization': False,
            'feature_map': True if model in ['LRCS8', 'LRCS9', 'LRCS10', 'Unet3'] else False,
        }

        graph_def_dir = re.search('(.+)ckpt/step\d+\.meta', ckpt_path).group(1)
        steps = args.steps
        for step in steps:
            paths = {
                'step': step,
                'perplexity': 100,  #default 30 usual range 5-50
                'niter': 5000,  #default 5000
                'working_dir': graph_def_dir,
                'ckpt_dir': graph_def_dir + 'ckpt/',
                'ckpt_path': graph_def_dir + 'ckpt/step{}'.format(step),
                'save_pb_dir': graph_def_dir + 'pb/',
                'save_pb_path': graph_def_dir + 'pb/step{}.pb'.format(step),
                'data_dir': args.data_dir,
                'rlt_dir':  graph_def_dir + 'rlt/',
                'tsne_dir':  graph_def_dir + 'tsne/',
                'tsne_path':  graph_def_dir + 'tsne/',
            }
            print(args.data_dir)
            conserve_nodes = args.conserve_nodes

            logger.info('paths: {}'.format(paths))
            logger.info('hyper: {}'.format(hyperparams))
            for type in args.retrieve_type:
                if type == 'activation':
                    partialRlt_and_diff(paths=paths, hyperparams=hyperparams, conserve_nodes=conserve_nodes)
                elif type == 'weight':
                    visualize_weights(params=paths)
                elif type == 'tsne':
                    tsne_on_weights(params=paths, mode='2D')
                    tsne_on_bias(params=paths, mode='2D')
                elif type == 'l2':
                    weights_euclidean_distance(ckpt_dir=paths['ckpt_dir'], rlt_dir=paths['rlt_dir'])
                elif type == 'ang':
                    weights_angularity(ckpt_dir=paths['ckpt_dir'], rlt_dir=paths['rlt_dir'])
                elif type == 'hist':
                    weights_hists_2excel(ckpt_dir=paths['ckpt_dir'], rlt_dir=paths['rlt_dir'])
                else:
                    logger.warning('%s is not an expected type' % type)

