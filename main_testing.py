from evaluate import testing

# force this to run on main
if __name__ == '__main__':
    d = './logs/2020_5_10_bs8_ps512_lrprogrammed_cs3_nc32_do0.0_act_leaky_aug_True_BN_True_mdl_LRCS11_mode_classification_lossFn_DSC_rampdecay0.0001_k0.3_p1.0_comment__GT_more_pore_here/hour23_gpu0/ckpt/'
    save_pb_dir = '/'.join(d.split('/')[:-2]) + '/pb/'

    paths = {
        'ckpt_dir': d,
        'label_dir': './testdata/',
        'save_pb_dir': save_pb_dir,
        'working_dir': '/'.join(d.split('/')[:-1]) + '/',
    }

    hyperparams = {
        'mode': 'classification',
        'feature_map': False,
        'loss_option': 'DSC',
        'batch_normalization': True,
        'device_option': 'cpu'
    }

    testing(paths=paths,
            hyper=hyperparams,
            numpy=True
            )