import os

l_nc = [32]
l_cs = [3]
l_lr = ['ramp', 'ramp', 'ramp']  #1e-5, 'ramp', 'exp'
init_lr = [1e-4, 1e-4, 1e-4]
k = [0.5, 0.5, 0.3]
period = [1, 1, 1]   #exp: k=1e-5 strong decay after 4 epoch ramp: 0.5
l_BN = [True]
l_do = [0.1]

for _do in l_do:
    for _BN in l_BN:
        for _lr, _init_lr, _k, _p in zip(l_lr, init_lr, k, period):
            for _cs in l_cs:
                for _nc in l_nc:
                    os.system('python ./main_train.py -nc {} -bs {} -ws {} -ep {} -cs {} -lr {} -ilr {} -klr {} -plr {} -bn {} -do {} -ag {} -fn {} -af {} -mdl {} -mode {}'.format(_nc, 8, 512, 5, _cs, _lr, _init_lr, _k, _p, _BN, 0.1, True, 'DSC', 'leaky', 'LRCS', 'classification'))


