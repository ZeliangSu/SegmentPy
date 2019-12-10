import tensorflow as tf
from input import inputpipeline
import datetime
import os
from layers import *
from tqdm import tqdm
import numpy as np
import logging
import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.DEBUG)

# params
hyperparams = {
    'patch_size': 80,
    'batch_size': 300,  #Xlearn < 20, Unet < 20 saturate GPU memory
    'nb_epoch': 100,
    'nb_batch': None,
    'conv_size': 9,
    'nb_conv': 80,
    'learning_rate': 1e-3,  #float or np.array of programmed learning rate
    'dropout': 0.1,
    'date': '{}_{}_{}'.format(datetime.datetime.now().year, datetime.datetime.now().month, datetime.datetime.now().day),
    'hour': '{}'.format(datetime.datetime.now().hour),
    'device_option': 'specific_gpu:0',
    'augmentation': False,
    'activation': 'relu',
    'save_step': 1000,
    'folder_name': None,
}

hyperparams['resume_folder_name'] = './logs/2019_11_5_bs300_ps80_lr0.001_cs9_nc80_do0.1_act_relu_aug_True_commentLRCS_conv4bbdnn_leaky/hour17/'

# Xlearn
conserve_nodes = [
    'model/decoder/logits/identity',
]

# get list of file names
hyperparams['totrain_files'] = [os.path.join('./proc/train/{}/'.format(hyperparams['patch_size']),
                              f) for f in os.listdir('./proc/train/{}/'.format(hyperparams['patch_size'])) if f.endswith('.h5')]
hyperparams['totest_files'] = [os.path.join('./proc/test/{}/'.format(hyperparams['patch_size']),
                             f) for f in os.listdir('./proc/test/{}/'.format(hyperparams['patch_size'])) if f.endswith('.h5')]

# find where one stopped
ckpt_dir = hyperparams['resume_folder_name'] + 'ckpt/'
tmp = []
for fname in os.listdir(ckpt_dir):
    if fname.endswith('.meta'):
        tmp.append(int(fname.split('step')[1].split('.')[0]))
hyperparams['end_at_step'] = max(tmp)
hyperparams['ckpt_path'] = ckpt_dir + 'step{}'.format(hyperparams['end_at_step'])

# init input pipeline
train_inputs = inputpipeline(hyperparams['batch_size'], suffix='train', augmentation=hyperparams['augmentation'])
test_inputs = inputpipeline(hyperparams['batch_size'], suffix='test')

new_training_type = tf.placeholder(tf.string, name='new_training_type')
new_drop_prob = tf.placeholder(tf.float32, name='new_dropout_prob')
new_lr = tf.placeholder(tf.float32, name='new_learning_rate')

with tf.name_scope('resume_input_pipeline'):
    X_dyn_batsize = hyperparams['batch_size']
    def f1(): return train_inputs
    def f2(): return test_inputs
    inputs = tf.cond(tf.equal(new_training_type, 'new_test'), lambda: f2(), lambda: f1(), name='new_input_cond')

# reload model from checkpoint
restorer = tf.train.import_meta_graph(
            hyperparams['ckpt_path'] + '.meta',
            input_map={
                'input_pipeline/input_cond/Merge_1': inputs['img'],
                'dropout_prob': new_drop_prob,
                'learning_rate': new_lr,
                'training_type': new_training_type,
            },
            clear_devices=True
        )

# calculate nb_batch
hyperparams['nb_batch'] = len(hyperparams['totrain_files']) // hyperparams['batch_size']

# get logit
input_graph = tf.get_default_graph()
input_graph_def = input_graph.as_graph_def()
new_logit = input_graph.get_tensor_by_name(conserve_nodes[0] + ':0')

# construct operation
with tf.name_scope('new_operation'):
    # optimizer/train operation
    mse = loss_fn(inputs['label'], new_logit, name='new_loss_fn')
    # https://github.com/tensorflow/tensorflow/issues/30017#issuecomment-522228763
    opt = optimizer(new_lr, name='new_optimizeR')

    # program gradients
    grads = opt.compute_gradients(mse)
    grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])

    # visualize graph in Board
    tf.summary.FileWriter('./dummy/graph/', tf.get_default_graph())

    # train operation
    def f3():
        return opt.apply_gradients(grads, name='new_train_op')
    def f4():
        return tf.no_op(name='new_no_op')
    train_op = tf.cond(tf.equal(new_training_type, 'train_type'), lambda: f3(), lambda: f4(), name='new_train_cond')

    with tf.name_scope('new_summary'):
        merged = tf.summary.merge([grad_sum])  # fixme: withdraw summary of histories for GPU resource reason

    with tf.name_scope('new_metrics'):
        m_loss, loss_up_op, m_acc, acc_up_op = metrics(new_logit, inputs['label'], mse, new_training_type)

# clean graph
tf.reset_default_graph()

# run train section
with tf.Session(graph=input_graph) as sess:
    restorer.restore(sess, hyperparams['ckpt_path'])

    # https://github.com/tensorflow/tensorflow/issues/30017#issuecomment-522228763
    uninitialized_vars = []
    for var in tf.all_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            uninitialized_vars.append(var)

    tf.initialize_variables(uninitialized_vars)

    # init summary
    folder = hyperparams['resume_folder_name']
    train_writer = tf.summary.FileWriter(folder + 'train/', sess.graph)
    cv_writer = tf.summary.FileWriter(folder + 'cv/', sess.graph)
    test_writer = tf.summary.FileWriter(folder + 'test/', sess.graph)

    saver = tf.train.Saver(max_to_keep=100000000)
    _globalStep = None
    try:
        for ep in tqdm(range(hyperparams['nb_epoch']), desc='Epoch'):

            # init ops
            sess.run(train_inputs['iterator_init_op'],
                     feed_dict={train_inputs['fnames_ph']: hyperparams['totrain_files'],
                                train_inputs['patch_size_ph']: [hyperparams['patch_size']] * len(
                                    hyperparams['totrain_files'])})
            sess.run(test_inputs['iterator_init_op'],
                     feed_dict={test_inputs['fnames_ph']: hyperparams['totest_files'],
                                test_inputs['patch_size_ph']: [hyperparams['patch_size']] * len(
                                    hyperparams['totest_files'])})

            # retrain
            for step in tqdm(range(hyperparams['nb_batch']), desc='Batch step'):
                if isinstance(hyperparams['learning_rate'], np.ndarray):
                    learning_rate = hyperparams['learning_rate'][ep * hyperparams['nb_batch'] + step]
                else:
                    learning_rate = hyperparams['learning_rate']
                try:
                    # note: 80%train 10%cross-validation 10%test
                    if step % 9 == 8:
                        # 10 percent of the data will be use to cross-validation
                        summary, _, _ = sess.run(
                            [merged, loss_up_op, acc_up_op],
                            feed_dict={new_training_type: 'cv',
                                       new_drop_prob: 1,
                                       new_lr: learning_rate})
                        cv_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)

                        # in situ testing without loading weights unlike cs-230-stanford
                        summary, _, _ = sess.run(
                            [merged, loss_up_op, acc_up_op],
                            feed_dict={new_training_type: 'test',
                                       new_drop_prob: 1,
                                       new_lr: learning_rate})
                        test_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)
                    else:
                        summary, _, _, _ = sess.run(
                            [merged, train_op, loss_up_op, acc_up_op],
                            feed_dict={new_training_type: 'train',
                                       new_drop_prob: hyperparams['dropout'],
                                       new_lr: learning_rate})
                        train_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)
                except tf.errors.OutOfRangeError as e:
                    print(e)
                    break

                if step % hyperparams['save_step'] == 0:
                    _globalStep = hyperparams['end_at_step'] + step + ep * hyperparams['nb_batch']
                    saver.save(sess, folder + 'ckpt/step{}'.format(_globalStep))
    except (KeyboardInterrupt, SystemExit):
        saver.save(sess, folder + 'ckpt/step{}'.format(_globalStep))
    saver.save(sess, folder + 'ckpt/step{}'.format(hyperparams['end_at_step'] + hyperparams['nb_epoch'] * hyperparams['nb_batch']))

