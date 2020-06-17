import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _roll(array, neib_value, shift=0, axis=0):
    _array = np.roll(array == neib_value, shift=shift, axis=axis)
    # Cancel the last slice rolled/shifted to the first slice
    if axis == 0:
        if shift >= 0:
            _array[:shift, :, :] = 0
        else:
            _array[shift:, :, :] = 0
        return _array
    elif axis == 1:
        if shift >= 0:
            _array[:, :shift, :] = 0
        else:
            _array[:, shift:, :] = 0
        return _array
    elif axis == 2:
        if shift >= 0:
            _array[:, :, :shift] = 0
        else:
            _array[:, :, -shift:] = 0
    return _array


def _roll_bis(array, that_value, shift=0, axis=0):
    # Roll the 3D array along axis with certain unity
    _array = np.roll(array != that_value, shift=shift, axis=axis)

    # Cancel the last slice rolled/shifted to the first slice
    if axis == 0:
        if shift >= 0:
            _array[:1, :, :] = 0
        else:
            _array[-1:, :, :] = 0
        return _array
    elif axis == 1:
        if shift >= 0:
            _array[:, :1, :] = 0
        else:
            _array[:, -1:, :] = 0
        return _array
    elif axis == 2:
        if shift >= 0:
            _array[:, :, :1] = 0
        else:
            _array[:, :, -1:] = 0
        return _array


def shift_helper(array, neib_value, shift1=0, axis1=0, shift2=0, axis2=0, shift3=0, axis3=0):
    # Roll the 3D array along axis with certain unity
    _array = _roll(_roll(_roll(array, neib_value, shift=shift1, axis=axis1),
                         1, shift=shift2, axis=axis2),
                   1, shift=shift3, axis=axis3)
    return _array


def shift_helper_bis(array, that_value, shift1=0, axis1=0, shift2=0, axis2=0, shift3=0, axis3=0):
    # Roll the 3D array along axis with certain unity
    _array = _roll_bis(_roll_bis(_roll_bis(array, that_value, shift=shift1, axis=axis1),
                         1, shift=shift2, axis=axis2),
                   1, shift=shift3, axis=axis3)
    return _array


def compneib26(array, phase1, phase2=None):
    if phase2 is not None:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                    & (shift_helper(array, phase2, shift1=-1, axis1=0)
                    | shift_helper(array, phase2, shift1=1, axis1=0)
                    | shift_helper(array, phase2, shift1=-1, axis1=1)
                    | shift_helper(array, phase2, shift1=1, axis1=1)
                    | shift_helper(array, phase2, shift1=-1, axis1=2)
                    | shift_helper(array, phase2, shift1=1, axis1=2)
                    | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=-1, axis2=1)
                    | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=1, axis2=1)
                    | shift_helper(array, phase2, shift1=1, axis1=0, shift2=-1, axis2=1)
                    | shift_helper(array, phase2, shift1=1, axis1=0, shift2=1, axis2=1)
                    | shift_helper(array, phase2, shift1=-1, axis1=1, shift2=-1, axis2=2)
                    | shift_helper(array, phase2, shift1=-1, axis1=1, shift2=1, axis2=2)
                    | shift_helper(array, phase2, shift1=1, axis1=1, shift2=-1, axis2=2)
                    | shift_helper(array, phase2, shift1=1, axis1=1, shift2=1, axis2=2)
                    | shift_helper(array, phase2, shift1=-1, axis1=2, shift2=-1, axis2=0)
                    | shift_helper(array, phase2, shift1=-1, axis1=2, shift2=1, axis2=0)
                    | shift_helper(array, phase2, shift1=-1, axis1=2, shift2=-1, axis2=0)
                    | shift_helper(array, phase2, shift1=1, axis1=2, shift2=1, axis2=0)
                    | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=-1, axis2=1, shift3=-1, axis3=2)
                    | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=-1, axis2=1, shift3=1, axis3=2)
                    | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=1, axis2=1, shift3=-1, axis3=2)
                    | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=1, axis2=1, shift3=1, axis3=2)
                    | shift_helper(array, phase2, shift1=1, axis1=0, shift2=-1, axis2=1, shift3=-1, axis3=2)
                    | shift_helper(array, phase2, shift1=1, axis1=0, shift2=-1, axis2=1, shift3=1, axis3=2)
                    | shift_helper(array, phase2, shift1=1, axis1=0, shift2=1, axis2=1, shift3=-1, axis3=2)
                    | shift_helper(array, phase2, shift1=1, axis1=0, shift2=1, axis2=1, shift3=1, axis3=2)
                    ))] = 1
    else:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                        & (shift_helper_bis(array, phase1, shift1=-1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=1)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=2)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=-1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=-1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=1, shift2=-1, axis2=2)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=1, shift2=1, axis2=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=1, shift2=-1, axis2=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=1, shift2=1, axis2=2)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=2, shift2=-1, axis2=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=2, shift2=1, axis2=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=2, shift2=-1, axis2=0)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=2, shift2=1, axis2=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=-1, axis2=1, shift3=-1, axis3=2)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=-1, axis2=1, shift3=1, axis3=2)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=1, axis2=1, shift3=-1, axis3=2)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=1, axis2=1, shift3=1, axis3=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=-1, axis2=1, shift3=-1, axis3=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=-1, axis2=1, shift3=1, axis3=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=1, axis2=1, shift3=-1, axis3=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=1, axis2=1, shift3=1, axis3=2)
                           ))] = 1
    return _array


def compneib18(array, phase1, phase2=None):
    if phase2 is not None:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                    & (shift_helper(array, phase2, shift1=-1, axis1=0)
                    | shift_helper(array, phase2, shift1=1, axis1=0)
                    | shift_helper(array, phase2, shift1=-1, axis1=1)
                    | shift_helper(array, phase2, shift1=1, axis1=1)
                    | shift_helper(array, phase2, shift1=-1, axis1=2)
                    | shift_helper(array, phase2, shift1=1, axis1=2)
                    | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=-1, axis2=1)
                    | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=1, axis2=1)
                    | shift_helper(array, phase2, shift1=1, axis1=0, shift2=-1, axis2=1)
                    | shift_helper(array, phase2, shift1=1, axis1=0, shift2=1, axis2=1)
                    | shift_helper(array, phase2, shift1=-1, axis1=1, shift2=-1, axis2=2)
                    | shift_helper(array, phase2, shift1=-1, axis1=1, shift2=1, axis2=2)
                    | shift_helper(array, phase2, shift1=1, axis1=1, shift2=-1, axis2=2)
                    | shift_helper(array, phase2, shift1=1, axis1=1, shift2=1, axis2=2)
                    | shift_helper(array, phase2, shift1=-1, axis1=2, shift2=-1, axis2=0)
                    | shift_helper(array, phase2, shift1=-1, axis1=2, shift2=1, axis2=0)
                    | shift_helper(array, phase2, shift1=-1, axis1=2, shift2=-1, axis2=0)
                    | shift_helper(array, phase2, shift1=1, axis1=2, shift2=1, axis2=0)
                    ))] = 1
    else:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                        & (shift_helper_bis(array, phase1, shift1=-1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=1)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=2)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=-1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=-1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=1, shift2=-1, axis2=2)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=1, shift2=1, axis2=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=1, shift2=-1, axis2=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=1, shift2=1, axis2=2)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=2, shift2=-1, axis2=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=2, shift2=1, axis2=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=2, shift2=-1, axis2=0)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=2, shift2=1, axis2=0)
                           ))] = 1
    return _array


def compneib8(array, phase1, phase2=None):
    if phase2 is not None:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                        & (shift_helper(array, phase2, shift1=-1, axis1=0)
                           | shift_helper(array, phase2, shift1=1, axis1=0)
                           | shift_helper(array, phase2, shift1=-1, axis1=1)
                           | shift_helper(array, phase2, shift1=1, axis1=1)
                           | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=-1, axis2=1)
                           | shift_helper(array, phase2, shift1=-1, axis1=0, shift2=1, axis2=1)
                           | shift_helper(array, phase2, shift1=1, axis1=0, shift2=-1, axis2=1)
                           | shift_helper(array, phase2, shift1=1, axis1=0, shift2=1, axis2=1)
                    ))] = 1
    else:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                        & (shift_helper_bis(array, phase1, shift1=-1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=1)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=-1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=0, shift2=1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=-1, axis2=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0, shift2=1, axis2=1)
                           ))] = 1
    return _array


def compneib6(array, phase1, phase2=None):
    if phase2 is not None:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                    & (shift_helper(array, phase2, shift1=-1, axis1=0)
                    | shift_helper(array, phase2, shift1=1, axis1=0)
                    | shift_helper(array, phase2, shift1=-1, axis1=1)
                    | shift_helper(array, phase2, shift1=1, axis1=1)
                    | shift_helper(array, phase2, shift1=-1, axis1=2)
                    | shift_helper(array, phase2, shift1=1, axis1=2)
                    ))] = 1
    else:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                        & (shift_helper_bis(array, phase1, shift1=-1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=1)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=2)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=2)
                           ))] = 1
    return _array


def compneib4(array, phase1, phase2=None):
    if phase2 is not None:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                    & (shift_helper(array, phase2, shift1=-1, axis1=0)
                    | shift_helper(array, phase2, shift1=1, axis1=0)
                    | shift_helper(array, phase2, shift1=-1, axis1=1)
                    | shift_helper(array, phase2, shift1=1, axis1=1)
                    ))] = 1
    else:
        _array = np.zeros(array.shape)
        _array[np.where((array == phase1)
                        & (shift_helper_bis(array, phase1, shift1=-1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=0)
                           | shift_helper_bis(array, phase1, shift1=-1, axis1=1)
                           | shift_helper_bis(array, phase1, shift1=1, axis1=1)
                           ))] = 1
    return _array


def choose_nb_neighb(mode, array, mtx):
    if mode == 6:
        _array = compneib6(array, **mtx)
    elif mode == 18:
        _array = compneib18(array, **mtx)
    elif mode == 26:
        _array = compneib26(array, **mtx)
    elif mode == 4:  # 2D
        _array = compneib4(array, **mtx)
    elif mode == 8:  # 2D
        _array = compneib8(array, **mtx)
    else:
        raise ValueError
    return _array


def get_surface(seg, phase: int):
    assert isinstance(seg, np.ndarray)
    surf = compneib8(seg, phase)
    return surf


def get_surface_3D(seg, phase:int):
    assert isinstance(seg, np.ndarray)
    assert len(seg.shape) == 3
    surf = compneib26(seg, phase)
    return surf


def get_interface(seg, phase1: int, phase2: int):
    assert isinstance(seg, np.ndarray)
    interface = compneib8(seg, phase1, phase2)
    return interface


def get_interface_3D(seg, phase1: int, phase2: int):
    assert isinstance(seg, np.ndarray)
    interface = compneib26(seg, phase1, phase2)
    return interface


def get_diff_map(seg, ground_truth):
    assert isinstance(seg, np.ndarray)
    assert isinstance(ground_truth, np.ndarray)
    assert seg.shape == ground_truth.shape
    diff = np.zeros_like(seg)
    diff[np.where(seg != ground_truth)] = 1
    return diff


def one_hot_2D(img):
    assert isinstance(img, np.ndarray)
    img = img.astype(np.int32)
    # get how many classes
    nb_classes = len(np.unique(img))
    # one hot
    out = []
    for i in range(nb_classes):
        tmp = np.zeros(img.shape)
        tmp[np.where(img == i)] = 1
        # tmp[np.where(tensor == i)] = 5  # uncomment this line to do 5-hot
        out.append(tmp)
    # stack along the last channel
    out = np.stack(out, axis=2).astype(np.int)
    return out


def IoU(seg, ground_truth):
    # get the series
    assert isinstance(seg, np.ndarray)
    assert isinstance(ground_truth, np.ndarray)
    assert seg.shape == ground_truth.shape  # one channel with 0, 1, 2

    _IoU = {}
    hotted_seg = one_hot_2D(seg)  # (h, w, cls)
    hotted_gt = one_hot_2D(ground_truth)  # (h, w, cls)

    tmp = hotted_gt * hotted_seg
    tmp2 = hotted_gt + hotted_seg

    for cls in np.unique(seg):
        _IoU[cls] = len(np.where(tmp[:, :, cls] != 0)[0]) / len(np.where(tmp2[:, :, cls] != 0)[0])
    return _IoU


def DSC(seg, ground_truth):
    '''
    Calculte Dice Similarity Coefficient between the segmented vol and the ground truth vol
    :param seg: 3D np ndarray of a segmented tomographic volume with different classes e.g. 0, 1, 2, ...
    :param ground_truth: 3D np ndaaray of a ground truth tomographic volume with different classes e.g. 0, 1, 2, ...
    :return: the value of Dice Similarity Coefficient of two inputs volumes
    '''
    # todo: only 2D version is used here considering it's hard to generate 3D ground truths

    # get the series
    assert isinstance(seg, np.ndarray)
    assert isinstance(ground_truth, np.ndarray)
    assert len(seg.shape) == len(ground_truth.shape)

    _DSC = {}
    hotted_seg = one_hot_2D(seg)  # (h, w, cls)
    hotted_gt = one_hot_2D(ground_truth)  # (h, w, cls)
    tp = 2 * np.product(hotted_seg, hotted_gt, axis=2) / np.sum(hotted_seg, hotted_gt, axis=2)
    for cls in np.unique(seg):
        _DSC[cls] = tp[cls]
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


def volume_fractions(vol: np.ndarray):
    vol_frac = {}
    for ph in np.unique(vol):
        vol_frac[ph] = len(np.where(vol == ph)[0]) / vol.size
    return vol_frac


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

    hyperparams = {
        'patch_size': 512,
        'batch_size': 8,
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

    # X_stack, y_stack, _ = _tifReader(paths['in_dir'])
    # #
    # # # RF inf
    # clf = load_model(model_path=paths['RF_model_path'])
    # vol_RF = predict(X_stack, clf, rlt_dir=paths['RF_out_dir'], filt_names=filt_names)
    # #
    # # # NN inf
    # vol_NN = inference_recursive(inputs=X_stack, conserve_nodes=conserve_nodes, paths=paths, hyper=hyperparams)
    # #
    # # # compute metrics
    # vol_RF = np.asarray(vol_RF)
    # vol_NN = np.asarray(vol_NN)
    #
    # # or read from folder
    # vol_RF, _, _ = _tifReader(paths['RF_out_dir'])
    # vol_NN, _, _ = _tifReader(paths['out_dir'])
    # X_stack, y_stack, _ = _tifReader(paths['in_dir'])
    # vol_RF = np.asarray(vol_RF)
    # vol_NN = np.asarray(vol_NN)
    # y_stack = np.asarray(y_stack)
    #
    # # or load from rlt
    # print('\nRF:', DSC(vol_RF, y_stack))
    # print('\nNN:', DSC(vol_NN, y_stack))
    # print('\nRF:', ACC(vol_RF, y_stack))
    # print('\nNN:', ACC(vol_NN, y_stack))
    #
    # # plot inverse histogram
    # # check path
    # check_N_mkdir(paths['rlt_dir'])
    # hist_inversing(X_stack, vol_RF, classes=['NMC', 'CBD', 'pore'], plot=True, rlt_path=paths['rlt_dir'] + 'RF.png')
    # hist_inversing(X_stack, vol_NN, classes=['NMC', 'CBD', 'pore'], plot=True, rlt_path=paths['rlt_dir'] + 'NN.png')

