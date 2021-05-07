import tensorflow as tf
import os

from tqdm import tqdm
import pandas as pd
import re

from model import *
from input import coords_gen
from util import check_N_mkdir
from input import inputpipeline_V2

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)


def main_train(
        hyperparams: dict,  # can be an object
        grad_view=False,
        nb_classes=3,
        resume=None,
    ):

    # clean graph exists in memory
    tf.reset_default_graph()

    # init input pipeline
    if hyperparams['model'] in ['LRCS8', 'LRCS9', 'LRCS10', 'Unet3']:
        print('**********************************Use multi channels input')
        train_inputs = inputpipeline_V2(hyperparams['batch_size'], hyperparams['patch_size'], suffix='train',
                                        augmentation=hyperparams['augmentation'], mode='weka')
        test_inputs = inputpipeline_V2(hyperparams['batch_size'], hyperparams['patch_size'],
                                       suffix='test', mode='weka')

    else:
        train_inputs = inputpipeline_V2(hyperparams['batch_size'], hyperparams['patch_size'], suffix='train',
                                        augmentation=hyperparams['augmentation'], mode='classification')
        test_inputs = inputpipeline_V2(hyperparams['batch_size'], hyperparams['patch_size'],
                                       suffix='test', mode='classification')

    # define other placeholder
    if hyperparams['dropout'] is not None:
        drop_prob = tf.placeholder(tf.float32, name='dropout_prob')
    else:
        drop_prob = tf.placeholder_with_default(1.0, [], name='dropout_prob')

    if hyperparams['batch_normalization']:
        BN_phase = tf.placeholder_with_default(False, (), name='BN_phase')
    else:
        BN_phase = False

    # init model
    lr = tf.placeholder(tf.float32, name='learning_rate')
    list_placeholders = [drop_prob, lr, BN_phase]
    train_nodes = classification_nodes(pipeline=train_inputs,
                                       placeholders=list_placeholders,
                                       model_name=hyperparams['model'],
                                       patch_size=hyperparams['patch_size'],
                                       batch_size=hyperparams['batch_size'],
                                       conv_size=hyperparams['conv_size'],
                                       nb_conv=hyperparams['nb_conv'],
                                       activation=hyperparams['activation'],
                                       batch_norm=hyperparams['batch_normalization'],
                                       loss_option=hyperparams['loss_option'],
                                       is_training=True,
                                       grad_view=grad_view,
                                       nb_classes=nb_classes,
                                       )

    test_nodes = classification_nodes(pipeline=test_inputs,
                                      placeholders=list_placeholders,
                                      model_name=hyperparams['model'],
                                      patch_size=hyperparams['patch_size'],
                                      batch_size=hyperparams['batch_size'],
                                      conv_size=hyperparams['conv_size'],
                                      nb_conv=hyperparams['nb_conv'],
                                      activation=hyperparams['activation'],
                                      batch_norm=hyperparams['batch_normalization'],
                                      loss_option=hyperparams['loss_option'],
                                      is_training=False,  # note: False: reuse the nodes from train_nodes but use different input_pipeline
                                      grad_view=False,
                                      nb_classes=nb_classes,
                                      )

    # print number of params
    print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

    # create logs folder
    check_N_mkdir(os.path.join(os.path.dirname(__file__), 'logs'))

    # start training/resume training
    if resume:
        _train_eval(train_nodes, test_nodes, train_inputs, test_inputs, hyperparams, resume=resume)
    else:
        _train_eval(train_nodes, test_nodes, train_inputs, test_inputs, hyperparams)


def _train_eval(train_nodes, test_nodes, train_inputs, test_inputs, hyperparams, resume=None):
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
    with tf.Session() as sess:
        # tf.summary.FileWriter('./dummy/debug/', sess.graph)
        # init params
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # init summary
        folder = hyperparams['folder_name']
        train_writer = tf.summary.FileWriter(os.path.join(folder, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(folder, 'test'), sess.graph)

        if not os.path.exists(os.path.join(folder, 'ckpt')):
            os.mkdir(os.path.join(folder, 'ckpt'))

        if resume:
            logger.info('The nodes to be restored: {}'.format(resume))
            assert 'from_ckpt' in hyperparams.keys(), 'be sure to have indicated from which ckpt to resume'
            assert isinstance(resume, list), 'resume arg should be a list'
            # get all trainable variable then retrieve the ones that want to restore
            all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            var_to_restore = []
            for var in all_var:
                if var.name in resume:
                    var_to_restore.append(var)
            saver = tf.train.Saver(var_to_restore, max_to_keep=5)
            best_saver = tf.train.Saver(var_to_restore, max_to_keep=1)

        else:
            # otherwise resume the whole network
            saver = tf.train.Saver(max_to_keep=5)
            best_saver = tf.train.Saver(max_to_keep=1)

        df = pd.DataFrame({'step': [0], 'val_acc': [0]})
        best_step = 0

        try:
            for ep in tqdm(range(hyperparams['nb_epoch']), desc='Epoch'):
                # init ops
                hyperparams['input_coords'].shuffle()
                ls_fname_train, ls_ps_train, ls_x_train, ls_y_train = hyperparams['input_coords'].get_train_args()
                ls_fname_test, ls_ps_test, ls_x_test, ls_y_test = hyperparams['input_coords'].get_valid_args()
                logger.debug(hyperparams)

                sess.run(train_inputs['iterator_init_op'],
                         feed_dict={train_inputs['fnames_ph']: ls_fname_train,
                                    train_inputs['patch_size_ph']: ls_ps_train,
                                    train_inputs['x_coord_ph']: ls_x_train,
                                    train_inputs['y_coord_ph']: ls_y_train,
                                    train_inputs['max_nb_cls_ph']: [hyperparams['max_nb_cls']] * hyperparams['nb_batch'],
                                    train_inputs['correction_ph']: [hyperparams['correction']] * hyperparams['nb_batch'],
                                    train_inputs['stretch_ph']: [hyperparams['stretch']] * hyperparams['nb_batch']
                                    })

                sess.run(test_inputs['iterator_init_op'],
                         feed_dict={test_inputs['fnames_ph']: ls_fname_test,
                                    test_inputs['patch_size_ph']: ls_ps_test,
                                    test_inputs['x_coord_ph']: ls_x_test,
                                    test_inputs['y_coord_ph']: ls_y_test,
                                    test_inputs['max_nb_cls_ph']: [hyperparams['max_nb_cls']] * len(ls_x_test),
                                    test_inputs['correction_ph']: [hyperparams['correction']] * len(ls_x_test),
                                    test_inputs['stretch_ph']: [hyperparams['stretch']] * len(ls_x_test)
                                    })

                # begin training
                for step in tqdm(range(hyperparams['nb_batch']), desc='Batch step'):
                    global_step = ep * hyperparams['nb_batch'] + step
                    learning_rate = hyperparams['learning_rate'][global_step]

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

                            train_writer.add_summary(summary, global_step)

                        else:
                            _, _, _, _ = sess.run([
                                train_nodes['train_op'], train_nodes['loss_update_op'], train_nodes['acc_update_op'], update_ops
                            ],
                                feed_dict=feed_dict)

                    except tf.errors.OutOfRangeError as e:
                        saver.save(sess, os.path.join(folder, 'ckpt', 'step{}'.format(global_step)))
                        print(e)
                        break

                    #save model
                    if global_step % hyperparams['save_step'] == 0:
                        saver.save(sess, os.path.join(folder, 'ckpt', 'step{}'.format(global_step)))
                        ########################
                        #
                        # valid session
                        #
                        ########################
                        # change feed dict
                        feed_dict = {
                            test_nodes['learning_rate']: 1.0,
                            test_nodes['drop']: 1.0,
                        }

                        if hyperparams['batch_normalization']:
                            feed_dict[test_nodes['BN_phase']] = False

                        for i_batch in tqdm(range(hyperparams['save_step'] // 10), desc='val batch'):
                            _, summary, _, _, val_acc = sess.run(
                                [
                                    test_nodes['y_pred'],
                                    test_nodes['summary'],
                                    test_nodes['loss_update_op'],
                                    test_nodes['acc_update_op'],
                                    test_nodes['val_acc']
                                ],
                                feed_dict=feed_dict
                            )
                            if i_batch == hyperparams['save_step'] // 10 - 1:
                                test_writer.add_summary(summary, global_step)

                        logger.info(val_acc)
                        df = df.append({'step': [global_step], 'val_acc': val_acc}, ignore_index=True)
                        logger.info(df['val_acc'])

                        rolling_progress = (df['val_acc'].max() - df['val_acc'].rolling(10, min_periods=1).mean().iloc[-1])
                        logger.info(rolling_progress)
                        argmax = df['val_acc'].argmax()
                        if argmax != best_step:
                            best_step = argmax
                            check_N_mkdir(os.path.join(folder, 'curves'))
                            best_saver.save(sess, os.path.join(folder, 'curves', 'best_model'))
                        if rolling_progress <= hyperparams['condition']:
                            logger.info('early stopped')
                            check_N_mkdir(os.path.join(folder, 'curves'))
                            saver.save(sess, os.path.join(folder, 'ckpt', 'step{}'.format(global_step)))
                            df.to_csv(os.path.join(folder, 'curves', 'in_loop_val_acc.csv'))
                            return

        except (KeyboardInterrupt, SystemExit) as e:
            check_N_mkdir(os.path.join(folder, 'curves'))
            saver.save(sess, os.path.join(folder, 'ckpt', 'step{}'.format(global_step)))
            df.to_csv(os.path.join(folder, 'curves', 'in_loop_val_acc.csv'))
            raise e
        check_N_mkdir(os.path.join(folder, 'curves'))
        saver.save(sess, os.path.join(folder, 'ckpt', 'step{}'.format(global_step)))
        df.to_csv(os.path.join(folder, 'curves', 'in_loop_val_acc.csv'))



