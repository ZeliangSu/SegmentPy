import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np


def train(nodes, train_inputs, test_inputs, hyperparams, save_step=200, device_option=None):
    """
    input:
    -------
        nodes: (tf.Graph?) model to train
        train_inputs: (tf.Iterator?) input of the trainning set inputpipeline
        test_inputs: (tf.Iterator?) input of the testing set inputpipeline
        hyperparams: (dictionary) dictionary that regroup all params
        save_step: (int) every such steps, save the weights, bias of the model
        device_option: (string) allow to choose on which GPU run the trainning or only on cpu
        e.g. 'specific_gpu:2' will run training on 3th gpu || 'cpu' will run only on cpu || None run on the 1st GPU
        mode: #TODO: running this mode will save CPU/GPU profiling into a .jason file. One can use Google Chrome to visualize the timeline occupancy of cpu and gpu
    return:
    -------
        None
    """
    # begin session
    config_params = {}
    if device_option == 'cpu':
        config_params['config'] = tf.ConfigProto(device_count={'GPU': 0})
    elif 'specific' in device_option:
        print('using GPU:{}'.format(device_option.split(':')[-1]))
        config_params['config'] = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=device_option.split(':')[-1]),
                                                 allow_soft_placement=True,
                                                 log_device_placement=False,
                                                 )

    with tf.Session(**config_params) as sess:
        # init params
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # init summary
        folder = hyperparams['folder_name']
        train_writer = tf.summary.FileWriter(folder + 'train/', sess.graph)
        cv_writer = tf.summary.FileWriter(folder + 'cv/', sess.graph)
        test_writer = tf.summary.FileWriter(folder + 'test/', sess.graph)

        if not os.path.exists(folder + 'ckpt/'):
            os.mkdir(folder + 'ckpt/')

        saver = tf.train.Saver(max_to_keep=100000000)
        _globalStep = None
        try:
            for ep in tqdm(range(hyperparams['nb_epoch']), desc='Epoch'):  # fixme: tqdm print new line after an exception
                # init ops
                sess.run(train_inputs['iterator_init_op'],
                         feed_dict={train_inputs['fnames_ph']: hyperparams['totrain_files'],
                                    train_inputs['patch_size_ph']: [hyperparams['patch_size']] * len(
                                        hyperparams['totrain_files'])})
                sess.run(test_inputs['iterator_init_op'],
                         feed_dict={test_inputs['fnames_ph']: hyperparams['totest_files'],
                                    test_inputs['patch_size_ph']: [hyperparams['patch_size']] * len(
                                        hyperparams['totest_files'])})

                # begin training
                for step in tqdm(range(hyperparams['nb_batch']), desc='Batch step'):
                    if isinstance(hyperparams['learning_rate'], np.ndarray):
                        learning_rate = hyperparams['learning_rate'][ep * hyperparams['nb_batch'] + step]
                    else:
                        learning_rate = hyperparams['learning_rate']
                    try:
                        #note: 80%train 10%cross-validation 10%test
                        if step % 9 == 8:
                            # 10 percent of the data will be use to cross-validation
                            summary, _, _, _ = sess.run([nodes['summary'], nodes['y_pred'], nodes['loss_update_op'], nodes['acc_update_op']],
                                                  feed_dict={nodes['train_or_test']: 'cv',
                                                             nodes['drop']: 1,
                                                             nodes['learning_rate']: learning_rate,
                                                             nodes['BN_phase']: False,
                            })
                            cv_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)

                        if step % 9 == 4: #note: (solution) unknown cross-validation nan-in-summary bug on batch norm
                            # in situ testing without loading weights unlike cs-230-stanford
                            summary, _, _, _ = sess.run([nodes['summary'], nodes['y_pred'], nodes['loss_update_op'], nodes['acc_update_op']],
                                                  feed_dict={nodes['train_or_test']: 'test',
                                                             nodes['drop']: 1,
                                                             nodes['learning_rate']: learning_rate,
                                                             nodes['BN_phase']: False, #fixme: if cv and test False successively throw nan-in-summary error
                                                             })
                            test_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)

                        else:
                            summary, _, _, _, _ = sess.run([
                                nodes['summary'], tf.get_collection(tf.GraphKeys.UPDATE_OPS),
                                nodes['train_op'], nodes['loss_update_op'], nodes['acc_update_op']
                            ],
                                feed_dict={nodes['train_or_test']: 'train',
                                           nodes['drop']: hyperparams['dropout'],
                                           nodes['learning_rate']: learning_rate,
                                           nodes['BN_phase']: True,
                                           })
                            train_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)
                    except tf.errors.OutOfRangeError as e:
                        print(e)
                        break

                    #save model
                    if step % save_step == 0:
                        _globalStep = step + ep * hyperparams['nb_batch']
                        saver.save(sess, folder + 'ckpt/step{}'.format(_globalStep))
        except (KeyboardInterrupt, SystemExit):
            saver.save(sess, folder + 'ckpt/step{}'.format(_globalStep))
        saver.save(sess, folder + 'ckpt/step{}'.format(hyperparams['nb_epoch'] * hyperparams['nb_batch']))


