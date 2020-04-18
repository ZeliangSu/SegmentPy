import tensorflow as tf
from tqdm import tqdm
import os
import numpy as np


def resume_training():
    pass


def train_1_epoch():
    pass


def test_1_epoch():
    pass


def train_test(train_nodes, test_nodes, train_inputs, test_inputs, hyperparams):
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
    #######################
    #
    # train session
    #
    #######################

    # init list
    summary_saved_at = []
    model_saved_at = []
    _step = None
    _ep = None

    with tf.Session() as sess:
        tf.summary.FileWriter('./dummy/debug/', sess.graph)
        # init params
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # init summary
        folder = hyperparams['folder_name']
        train_writer = tf.summary.FileWriter(folder + 'train/', sess.graph)
        test_writer = tf.summary.FileWriter(folder + 'test/', sess.graph)

        if not os.path.exists(folder + 'ckpt/'):
            os.mkdir(folder + 'ckpt/')

        saver = tf.train.Saver(max_to_keep=100000000)
        _globalStep = None
        try:
            for ep in tqdm(range(hyperparams['nb_epoch']), desc='Epoch'):  # fixme: tqdm print new line after an exception
                _ep = ep
                # init ops
                hyperparams['input_coords'].shuffle()
                ls_fname_train, ls_ps_train, ls_x_train, ls_y_train = hyperparams['input_coords'].get_train_args()
                ls_fname_test, ls_ps_test, ls_x_test, ls_y_test = hyperparams['input_coords'].get_test_args()
                sess.run(train_inputs['iterator_init_op'],
                         feed_dict={train_inputs['fnames_ph']: ls_fname_train,
                                    train_inputs['patch_size_ph']: ls_ps_train,
                                    train_inputs['x_coord_ph']: ls_x_train,
                                    train_inputs['y_coord_ph']: ls_y_train})
                sess.run(test_inputs['iterator_init_op'],
                         feed_dict={test_inputs['fnames_ph']: ls_fname_test,
                                    test_inputs['patch_size_ph']: ls_ps_test,
                                    test_inputs['x_coord_ph']: ls_x_test,
                                    test_inputs['y_coord_ph']: ls_y_test})

                # begin training
                for step in tqdm(range(hyperparams['nb_batch']), desc='Batch step'):
                    _step = step
                    learning_rate = hyperparams['learning_rate'][_ep * hyperparams['nb_batch'] + _step]

                    # for batch norm
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    # train
                    try:
                        # save summary
                        # distinguish if use BN and DO
                        feed_dict = {
                            train_nodes['learning_rate']: learning_rate,
                            train_nodes['drop']: hyperparams['dropout'],
                        }

                        if hyperparams['batch_normalization']:
                            feed_dict[train_nodes['BN_phase']] = True

                        if step % hyperparams['save_summary_step'] == 0:
                            _, _, _, summary, _ = sess.run([
                                train_nodes['train_op'], train_nodes['loss_update_op'], train_nodes['acc_update_op'],
                                train_nodes['summary'], update_ops,
                            ],
                                feed_dict=feed_dict)

                            train_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)
                            summary_saved_at.append(step)
                        else:
                            _, _, _, _ = sess.run([
                                train_nodes['train_op'], train_nodes['loss_update_op'], train_nodes['acc_update_op'], update_ops
                            ],
                                feed_dict=feed_dict)

                        #note: learning rate soucis
                        # print(sess.run(train_nodes['learning_rate'], feed_dict=feed_dict))

                    except tf.errors.OutOfRangeError as e:
                        saver.save(sess, folder + 'ckpt/step{}'.format(_globalStep))
                        model_saved_at.append(step)
                        print(e)
                        break


                    #save model
                    if step % hyperparams['save_step'] == 0:
                        saver.save(sess, folder + 'ckpt/step{}'.format(_ep * hyperparams['nb_batch'] + _step))
                        model_saved_at.append(step)

                        ########################
                        #
                        # test session
                        #
                        ########################
                        if step != 0:
                            # change feed dict
                            feed_dict = {
                                train_nodes['learning_rate']: 1.0,
                                train_nodes['drop']: 1.0,
                            }

                            if hyperparams['batch_normalization']:
                                feed_dict[train_nodes['BN_phase']] = False

                            # load graph in the second device
                            loader = tf.train.Saver()
                            # change
                            ckpt_saved_path = folder + 'ckpt/step{}'.format(_ep * hyperparams['nb_batch'] + _step)
                            loader.restore(sess, ckpt_saved_path)
                            for i_batch in tqdm(range(hyperparams['save_step'] // 10), desc='test batch'):
                                _, summary, _, _ = sess.run(
                                    [
                                        test_nodes['y_pred'],
                                        test_nodes['summary'],
                                        test_nodes['loss_update_op'],
                                        test_nodes['acc_update_op']
                                    ],
                                    feed_dict=feed_dict
                                )
                                if i_batch == hyperparams['save_step'] // 10 - 1:
                                    test_writer.add_summary(summary, _ep * hyperparams['nb_batch'] + _step)

        except (KeyboardInterrupt, SystemExit):
            saver.save(sess, folder + 'ckpt/step{}'.format(_ep * hyperparams['nb_batch'] + _step))
        saver.save(sess, folder + 'ckpt/step{}'.format(_ep * hyperparams['nb_batch'] + _step))



