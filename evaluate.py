import tensorflow as tf
from tqdm import tqdm
from util import list_ckpts, load_img, get_img_stack, read_pb
from layers import DSC, Cross_Entropy, DSC_np, Cross_Entropy_np, customized_softmax, customized_softmax_np
from inference import freeze_ckpt_for_inference, optimize_pb_for_inference
from input import _one_hot, _inverse_one_hot
import numpy as np
import re
import os

# logging
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)

e = 'https://github.com/tensorflow/tensorflow/issues/31318'
# todo: note:
logger.warning('Loading the following function with those in train.py will cause GPU inferences similar to : {}'.format(e))


def _get_nodes(graph_def: tf.GraphDef, out_node: list, hyper: dict, if_numpy=True):

    # overwrite/get i/o
    _ = tf.import_graph_def(graph_def, return_elements=[out_node[-1]])
    Graph = tf.get_default_graph()
    new_input = Graph.get_tensor_by_name('import/new_input:0')
    y = Graph.get_tensor_by_name('import/' + out_node[-1] + ':0')
    bn = Graph.get_tensor_by_name('import/new_BN:0')

    if if_numpy:
        return {
            'new_input': new_input,
            'new_BN_phase': bn,
            'y_hat': y,
        }

    else:
        # todo: the following might be simplified
        new_label = tf.placeholder(tf.int32, shape=[None, None, None, 10 if hyper['feature_map'] else 1],
                                   name='new_label')

        # graffe new loss/acc part
        y = customized_softmax(y)
        y = tf.argmax(y, axis=3)
        if hyper['loss_option'] == 'Dice':
            loss = DSC(new_label, y, name='loss_fn')
        elif hyper['loss_option'] == 'cross_entropy':
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
            'acc_val_op': acc_val_op,
            'lss_update_op': loss_update_op,
            'lss_val_op': loss_val_op,
            'summary': merged
        }


def testing_recursive(paths: dict, hyper: dict, numpy: bool):
    # evaluate in cpu to avoid the training in gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # prepare test labels
    img = get_img_stack(paths['label_dir'], img_or_label='input')
    label = get_img_stack(paths['label_dir'], img_or_label='label')

    # misc
    l_ckpt_p = list_ckpts(paths['ckpt_dir'])[1]
    mdl_name = re.search('mdl\_([A-Za-z]+\d*)', l_ckpt_p[0]).group(1)
    out_node = ['{}/decoder/logits/identity'.format(mdl_name)]

    # note: here could not restore directly from the ckpt because of the pipeline, need to be frozon and optimized
    #  or should refer to landscape
    for step in tqdm(list_ckpts(paths['ckpt_dir'])[0]):
        # define some paths
        paths['save_pb_path'] = paths['save_pb_dir'] + 'frozen_step{}.pb'.format(step)
        paths['optimized_pb_path'] = paths['save_pb_dir'] + 'optimized_step{}.pb'.format(step)
        paths['ckpt_path'] = paths['ckpt_dir'] + 'step{}'.format(step)
        if not os.path.exists(paths['optimized_pb_path']):
            # freeze to pb
            freeze_ckpt_for_inference(paths=paths, hyper=hyper,
                                      conserve_nodes=out_node)  # there's still some residual nodes
            optimize_pb_for_inference(paths=paths,
                                      conserve_nodes=out_node)

    # inference and evaluate
    accs = []
    lsss = []
    steps = []
    for step in tqdm(list_ckpts(paths['ckpt_dir'])[0]):
        tf.reset_default_graph()  # note: should clean the graph before reloading another pb or the params will not change
        with tf.Session() as sess:
            # new summary writer
            test_writer = tf.summary.FileWriter('/'.join(paths['ckpt_dir'].split('/')[:-2]) + '/' + 'new_test/', sess.graph)

            pb_path = paths['save_pb_dir'] + 'optimized_step{}.pb'.format(step)
            graph_def_optimized = read_pb(pb_path)

            # get nodes
            nodes = _get_nodes(graph_def=graph_def_optimized, out_node=out_node, hyper=hyper, if_numpy=numpy)

            # load params and evaluate
            logger.debug(step)
            acc, lss = _evaluate(
                sess=sess,
                writer=test_writer,
                global_step=step,
                nodes=nodes,
                img=img,
                label=label
            )
            accs.append(acc)
            lsss.append(lss)
            steps.append(step)
    ds = np.stack([steps, accs, lsss], axis=0)
    np.savetxt(fname='/'.join(paths['ckpt_dir'].split('/')[:-2]) + '/' + 'new_test/new_curve.csv', X=ds.transpose())


def testing(paths: dict, hyper: dict, numpy: bool):
    # evaluate in cpu to avoid the training in gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # prepare test labels
    img = get_img_stack(paths['test_dir'], img_or_label='input') * hyper['norm']
    label = get_img_stack(paths['test_dir'], img_or_label='label')
    logger.debug('label: {} unique: {}'.format(label.shape, np.unique(label)))

    # misc
    ckpt = paths['ckpt_path']
    mdl_name = re.search('mdl\_([A-Za-z]+\d*)', ckpt).group(1)
    step = re.search('step(\d+)', ckpt).group(1)
    step = int(step)
    out_node = ['{}/decoder/logits/identity'.format(mdl_name)]

    # note: here could not restore directly from the ckpt because of the pipeline, need to be frozon and optimized
    #  or should refer to landscape

    # define some paths
    paths['save_pb_path'] = paths['save_pb_dir'] + 'frozen_step{}.pb'.format(step)
    paths['optimized_pb_path'] = paths['save_pb_dir'] + 'optimized_step{}.pb'.format(step)
    if not os.path.exists(paths['optimized_pb_path']):
        # freeze to pb
        freeze_ckpt_for_inference(paths=paths, hyper=hyper,
                                  conserve_nodes=out_node)  # there's still some residual nodes
        optimize_pb_for_inference(paths=paths,
                                  conserve_nodes=out_node)

    # inference and evaluate

    tf.reset_default_graph()  # note: should clean the graph before reloading another pb or the params will not change
    with tf.Session() as sess:
        graph_def_optimized = read_pb(paths['save_pb_path'])

        # get nodes
        nodes = _get_nodes(graph_def=graph_def_optimized, out_node=out_node, hyper=hyper, if_numpy=numpy)

        # load params and evaluate
        logger.debug(step)
        acc, lss, y, label = _evaluate(
            sess=sess,
            global_step=step,
            nodes=nodes,
            img=img,
            label=label,
            numpy=True
        )
    return acc, lss, y, label


def _evaluate(sess: tf.Session,
              global_step: int,
              nodes: dict,
              img: np.ndarray,
              label: np.ndarray,
              writer=None,
              numpy=True,
             ):
    # init variables
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    # get metrics
    if numpy:
        feed_dict = {
            nodes['new_input']: img.reshape((-1, *img.shape[1:], 1)),
            nodes['new_BN_phase']: True,  # note: (TF114) here should be True while testing
        }
        logger.debug('label: {} {}'.format(label.shape, np.unique(label)))
        label = label.reshape((-1, *img.shape[1:], 1)).astype(np.int32)
        label = _one_hot(label)
        logger.debug('label: {} {}'.format(label.shape, np.unique(label)))

        y, = sess.run([nodes['y_hat']], feed_dict=feed_dict)
        y = customized_softmax_np(y)

        lss = DSC_np(
            y_true=label,  # [B, W, H, 3]
            logits=y  # [B, W, H, 3]
        )

        # inverse one hot then accuracy
        y = _inverse_one_hot(y)
        label = _inverse_one_hot(label)
        logger.debug('label: {} {}, y: {} {}'.format(label.shape, np.unique(label), y.shape, np.unique(y)))

        acc = len(np.where(y == label)[0]) / y.size
        logger.debug('lss: {}, acc: {} input shape: {} y shape:{}'.format(lss, acc, img.shape, y.shape))
        return acc, lss, y, label

    else:
        assert isinstance(writer, tf.summary.FileWriter)
        feed_dict = {
            nodes['new_input']: img.reshape((-1, *img.shape[1:], 1)),
            nodes['new_label']: label.reshape((-1, *img.shape[1:], 1)),
            nodes['new_BN_phase']: True,  # note: (TF114) here should be True while testing
        }
        for i in range(2):
            # fixme: (TF114) should have at least 2 iterations or acc=lss=0
            summary, _, _ = sess.run([nodes['summary'],
                                      nodes['acc_update_op'],
                                      nodes['lss_update_op'],
                                      ], feed_dict=feed_dict)

            # save summary
            writer.add_summary(summary, global_step=global_step)
        return sess.run([nodes['acc_val_op'], nodes['lss_val_op']], feed_dict=feed_dict)

