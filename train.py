import tensorflow as tf
from tqdm import tqdm
import os
from util import list_ckpts, load_img, get_img_stack
from layers import DSC, Cross_Entropy, customized_softmax
from tqdm import tqdm
import numpy as np
import re

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)


def resume_training():
    # re-define input pipeline's pyfunc
    # todo: might need TF2 as TF114 does not store pyfunc
    pass


def _graff_metrics(Graph: tf.Graph, out_node: str, hyper: dict):
    # todo: the following might be simplified
    # overwrite/get i/o
    # update_pipeline = Graph.get_operation_by_name('')
    new_input = Graph.get_tensor_by_name('new_input:0')
    y = Graph.get_tensor_by_name(out_node)
    bn = Graph.get_tensor_by_name('new_BN:0')

    # graffe new loss/acc part
    new_label = tf.placeholder(tf.int32, shape=[None, None, None, 10 if hyper['feature_map'] else 1],
                               name='new_label')
    if hyper['loss_option'] == 'DSC':
        y = customized_softmax(y)
        loss = DSC(new_label, y, name='loss_fn')
    elif hyper['loss_option'] == 'cross_entropy':
        y = customized_softmax(y)
        loss = Cross_Entropy(new_label, y, name='CE')
    else:
        raise NotImplementedError('Cannot find the loss option')

    # 1-hot
    if hyper['mode'] == 'classification':
        y = tf.cast(tf.argmax(y, axis=3), tf.int32)

    # redefine metrics
    loss_val_op, loss_update_op = tf.metrics.mean(loss, name='new_lss')
    acc_val_op, acc_update_op = tf.metrics.accuracy(labels=new_label, predictions=y, name='new_acc')
    merged = tf.summary.merge([tf.summary.scalar("new_lss", loss_val_op), tf.summary.scalar('new_acc', acc_val_op)])

    return {
        'new_input': new_input,
        'new_label': new_label,
        'new_BN_phase': bn,
        'acc_update_op': acc_update_op,
        'lss_update_op': loss_update_op,
        'summary': merged
    }


def re_test(ckpt_dir: str, label_dir: str, hyper: dict):
    # evaluate in cpu to avoid the training in gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # prepare test labels
    img = get_img_stack(label_dir, img_or_label='input')
    label = get_img_stack(label_dir, img_or_label='label')

    # make placeholder
    new_input = tf.placeholder(tf.float32, shape=[None, None, None, 10 if hyper['feature_map'] else 1], name='new_input')

    input_map = {'input_pipeline_test/IteratorGetNext': new_input}

    try:
        new_BN = tf.placeholder_with_default(False, [],
                                             name='new_BN')
        input_map['BN_phase'] = new_BN

    except Exception as e:
        print(e)
        pass
    try:
        new_dropout = tf.placeholder_with_default(1.0, [], name='new_dropout')
        input_map['dropout_prob'] = new_dropout

    except Exception as e:
        print(e)
        pass

    # inference and evaluate
    with tf.Session() as sess:
        l_ckpt_p = list_ckpts(ckpt_dir)[1]

        # restore graph
        restorer = tf.train.import_meta_graph(
            l_ckpt_p[0] + '.meta',
            input_map=input_map,
            clear_devices=True,
        )

        # misc
        mdl_name = re.search('mdl_(.*)_mode', l_ckpt_p[0]).group(1)
        out_node = '{}/decoder/logits/identity:0'.format(mdl_name)
        G = tf.get_default_graph()

        # get nodes
        nodes = _graff_metrics(Graph=G, out_node=out_node, hyper=hyper)

        # new summary writer
        test_writer = tf.summary.FileWriter('/'.join(ckpt_dir.split('/')[:-2]) + '/' + 'new_test/', sess.graph)

        # load params and evaluate
        for ckpt in tqdm(l_ckpt_p):
            logger.debug(ckpt)
            evaluate(
                sess=sess,
                loader=restorer,
                writer=test_writer,
                ckpt_path=ckpt,
                nodes=nodes,
                img=img,
                label=label
            )


def evaluate(sess: tf.Session,
             loader: tf.Graph,
             writer: tf.summary.FileWriter,
             ckpt_path: str,
             nodes: dict,
             img: np.ndarray,
             label: np.ndarray,
             ):

        # restore kernel params
        loader.restore(sess, ckpt_path)

        # init variables for metrics
        sess.run([tf.local_variables_initializer()])

        # prepare feed_dict
        feed_dict = {
            nodes['new_input']: img.reshape((-1, *img.shape[1:], 1)),
            nodes['new_label']: label.reshape((-1, *img.shape[1:], 1)),
            nodes['new_BN_phase']: False,
        }

        # get metrics
        summary, _, _ = sess.run([nodes['summary'], nodes['acc_update_op'], nodes['lss_update_op']], feed_dict=feed_dict)

        # save summary
        writer.add_summary(summary, global_step=re.search('step(\d+)', ckpt_path).group(1))


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
        try:
            for ep in tqdm(range(hyperparams['nb_epoch']), desc='Epoch'):  # fixme: tqdm print new line after an exception
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

                        #note: learning rate soucis
                        # print(sess.run(train_nodes['learning_rate'], feed_dict=feed_dict))

                    except tf.errors.OutOfRangeError as e:
                        saver.save(sess, folder + 'ckpt/step{}'.format(global_step))
                        # model_saved_at.append(step)
                        print(e)
                        break

                    #save model
                    if global_step % hyperparams['save_step'] == 0:
                        saver.save(sess, folder + 'ckpt/step{}'.format(global_step))
                        # model_saved_at.append(step)

                        ########################
                        #
                        # test session
                        #
                        ########################
                        # if step != 0:
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
                        ckpt_saved_path = folder + 'ckpt/step{}'.format(global_step)

                        # todo: this part is also load in gpu
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
                                test_writer.add_summary(summary, global_step)

        except (KeyboardInterrupt, SystemExit) as e:
            saver.save(sess, folder + 'ckpt/step{}'.format(global_step))
            raise e
        saver.save(sess, folder + 'ckpt/step{}'.format(global_step))



