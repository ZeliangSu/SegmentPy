import numpy as np
from reader import _tifReader
import pandas as pd
import matplotlib.pyplot as plt
from randomForest import load_model, predict
from inference import inference_recursive
from util import check_N_mkdir

class SU_series:
    def __init__(self, number):
        self.series = [0]
        self.sum = [0]
        _series = []
        _i = []
        _somme = []
        self.set_range(number)

    def set_range(self, range):
        self.range = range

    def make_series(self):
        for i in range(1, self.range):
            _series = self.series + [i]
            _i = [i]
            _somme = []
            for j in _series:
                _somme += [j + i]
            for k in _somme:
                if k in self.sum:
                    _somme = []
                    _i = []
                    break
            self.sum += _somme
            self.series += _i

    def get_series(self):
        self.make_series()
        return self.series


def DSC(seg, ground_truth):
    '''
    Calculte Dice Similarity Coefficient between the segmented vol and the ground truth vol
    :param seg: 3D np ndarray of a segmented tomographic volume with different classes e.g. 0, 1, 2, ...
    :param ground_truth: 3D np ndaaray of a ground truth tomographic volume with different classes e.g. 0, 1, 2, ...
    :return: the value of Dice Similarity Coefficient of two inputs volumes
    '''

    # get the series
    Series_a = SU_series(11).get_series()

    # init
    _DSC = {}

    # parse vols
    seg = seg.astype(int)
    seg_class = np.unique(seg)
    gt_class = np.unique(ground_truth)
    assert seg_class.all() == gt_class.all(), "There shouln't be different class! Seg:{}, Ground:{}'.format(seg_class, gt_class)"
    for i in range(seg_class.size):
        seg_class[i] = Series_a[i]  #0, 1, 3, 7...
        gt_class[i] = Series_a[i]   #0, 1, 3, 7...

    # convert 0,1,2,3 to 0,1,3,7 of the volumes
    for i in range(len(seg_class)):
        seg[np.where(seg == i)] = Series_a[i]
        ground_truth[np.where(ground_truth == i)] = Series_a[i]

    # sum 2 vols
    tmp = seg + ground_truth

    # calculate DSC
    if len(seg_class) <= 11:
        # uniques: [0, 1, 2, 3, 4, 6...]
        # counts: [T0, f01, T1, f03, f13, T3...]
        # dic: {0:T0, 1:f01, 2:T1, 3:f03, 4:f13, 6:T3...]
        uniques, counts = np.unique(tmp, return_counts=True)
        dic = dict(np.asarray((uniques, counts), dtype=int).T)

        # some init
        f_cls = 0

        # calculate DSC for each class
        for cls in seg_class:
            T_cls = dic[cls * 2]
            for rest in np.delete(seg_class, np.where(seg_class == cls)):
                f_cls += dic[cls + rest]

            # add DSC for each class in the dict
            # 2T_cls / (2T_cls + f_cls)
            _DSC[cls] = 2 * T_cls / (2 * T_cls + f_cls)
    else:
        raise NotImplementedError("only 11 classes(phases) are supported")
    return _DSC


def ACC(seg, ground_truth):
    '''
    :param seg:
    :param ground_truth:
    :return:
    '''
    assert seg.shape == ground_truth.shape, 'the volume and the GroundT shapes are differnent'
    seg = seg.astype(int)
    ACC = np.where(seg == ground_truth)[0].size / seg.size
    return ACC


def hist_inversing(init_arr, seg_arr, bins=2**10, classes=None, csv=None, plot=False, rlt_path=None):

    # convert from list to ndarray
    if isinstance(init_arr, list):
        init_arr = np.asarray(init_arr)
    if isinstance(seg_arr, list):
        seg_arr = np.asarray(seg_arr)
    seg_arr = seg_arr.astype(np.int)
    # inits
    elts = np.unique(seg_arr)
    if classes:
        assert isinstance(classes, list), 'only list is accepted for arg classes'
    # nb_class = elts.size
    df = {}
    arr_min, arr_max = init_arr.min(), init_arr.max()
    init_hist = np.histogram(init_arr, bins=bins, range=(arr_min, arr_max))
    df['x'] = init_hist[1]  # [1] for x axis
    df['init'] = init_hist[0]  # [0] for y axis

    # create graph
    i = 0
    for elt in elts:
        mask = np.zeros(seg_arr.shape)
        mask[np.where(seg_arr == elt)] = 1
        df['{}'.format(i)] = np.histogram(init_arr, bins=bins, range=(arr_min, arr_max), weights=mask)[0]
        i += 1
    if csv:
        pd.DataFrame(df).to_csv(csv)
    if plot:
        plt.figure('init vs hist')
        for k, v in df.items():
            if k == 'x':
                pass
            elif k == 'init':
                plt.plot(df['x'][:-1], df[k], label=k)
            else:
                plt.plot(df['x'][:-1], df[k], '.', label=k)
                # plt.fill_between(df['x'][:-1], df[k])  #fixme: color code issue
        plt.legend()
        plt.show()
        plt.savefig(rlt_path)


if __name__ == '__main__':

    conserve_nodes = [
            'model/decoder/logits/relu',
        ]

    filt_names = [
        'gaussian_blur',
        'sobel',
        'hessian',
        'dog',
        #'membrane_proj',
        'anisotropic_diffusion1',
        'anisotropic_diffusion2',
        'gabor',
        'bilateral',
        'median',
    ]

    hyperparams = {
        'patch_size': 80,
        'batch_size': None,
        'nb_batch': None,
        'nb_patch': None,
        'stride': 1,
        'device_option': 'specific_gpu:0',
    }
    _dir = './logs/2019_10_23_bs300_ps80_lr0.0001_cs7_nc80_do0.1_act_leaky_aug_True_commentConv4bb_1-leaky/hour16/'
    paths = {
        'step': 12864,
        'in_dir': './result/in/',
        'out_dir': './result/out/',
        'RF_out_dir': './RF_result/out/',
        'RF_model_path': './RF_result/mdl/model.sav',
        'working_dir': './logs/2019_10_23_bs300_ps80_lr0.0001_cs7_nc80_do0.1_act_leaky_aug_True_commentConv4bb_1-leaky/hour16/',
        'ckpt_dir': './logs/2019_10_21_bs300_ps80_lr0.0001_cs7_nc80_do0.1_act_leaky_aug_True/hour17/ckpt/',
        'ckpt_path': './logs/2019_10_21_bs300_ps80_lr0.0001_cs7_nc80_do0.1_act_leaky_aug_True/hour17/ckpt/step22728',
        'save_pb_dir': './logs/2019_10_21_bs300_ps80_lr0.0001_cs7_nc80_do0.1_act_leaky_aug_True/hour17/pb/',
        'save_pb_path': './logs/2019_10_21_bs300_ps80_lr0.0001_cs7_nc80_do0.1_act_leaky_aug_True/hour17/pb/frozen_step22728.pb',
        'optimized_pb_dir': './logs/2019_10_21_bs300_ps80_lr0.0001_cs7_nc80_do0.1_act_leaky_aug_True/hour17/optimize/',
        'optimized_pb_path': './logs/2019_10_21_bs300_ps80_lr0.0001_cs7_nc80_do0.1_act_leaky_aug_True/hour17/optimize/optimized_22728.pb',
        'rlt_dir': './logs/2019_10_21_bs300_ps80_lr0.0001_cs7_nc80_do0.1_act_leaky_aug_True/hour17/rlt/',
        'GPU': 0,
    }

    X_stack, y_stack, _ = _tifReader(paths['in_dir'])
    #
    # # RF inf
    clf = load_model(model_path=paths['RF_model_path'])
    vol_RF = predict(X_stack, clf, rlt_dir=paths['RF_out_dir'], filt_names=filt_names)
    #
    # # NN inf
    vol_NN = inference_recursive(inputs=X_stack, conserve_nodes=conserve_nodes, paths=paths, hyper=hyperparams)
    #
    # # compute metrics
    vol_RF = np.asarray(vol_RF)
    vol_NN = np.asarray(vol_NN)

    # or read from folder
    # vol_RF, _, _ = _tifReader(paths['RF_out_dir'])
    # vol_NN, _, _ = _tifReader(paths['out_dir'])
    # X_stack, y_stack, _ = _tifReader(paths['in_dir'])
    # vol_RF = np.asarray(vol_RF)
    # vol_NN = np.asarray(vol_NN)
    # y_stack = np.asarray(y_stack)

    # or load from rlt
    print('\nRF:', DSC(vol_RF, y_stack))
    print('\nNN:', DSC(vol_NN, y_stack))
    print('\nRF:', ACC(vol_RF, y_stack))
    print('\nNN:', ACC(vol_NN, y_stack))

    # plot inverse histogram
    # check path
    check_N_mkdir(paths['rlt_dir'])
    hist_inversing(X_stack, vol_RF, classes=['NMC', 'CBD', 'pore'], plot=True, rlt_path=paths['rlt_dir'] + 'RF.png')
    hist_inversing(X_stack, vol_NN, classes=['NMC', 'CBD', 'pore'], plot=True, rlt_path=paths['rlt_dir'] + 'NN.png')

