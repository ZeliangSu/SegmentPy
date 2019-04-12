import tensorflow as tf
from tqdm import tqdm

def train(nodes, train_inputs, test_inputs, hyperparams, save_step=200, device_option=None):
    # begin session
    config_params = {}
    if device_option == 'cpu':
        config_params['config'] = tf.ConfigProto(device_count={'GPU': 0})
    elif device_option == 'specific_gpu:1':
        config_params['config'] = tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=device_option.split(':')[-1]),
                                                 allow_soft_placement=True,
                                                 log_device_placement=False,
                                                 )

    with tf.Session(**config_params) as sess:

        # with tf.Session() as sess:
        # init params
        sess.run(tf.global_variables_initializer())

        # init summary
        train_writer = tf.summary.FileWriter('./logs/{}/hour{}/train/bs{}_ps{}_lr{}_cs{}'.format(hyperparams['date'],
                                                                                                 hyperparams['hour'],
                                                                                                 hyperparams['batch_size'],
                                                                                                 hyperparams['patch_size'],
                                                                                                 hyperparams['learning_rate'],
                                                                                                 hyperparams['conv_size']),
                                             sess.graph)
        cv_writer = tf.summary.FileWriter('./logs/{}/hour{}/cv/bs{}_ps{}_lr{}_cs{}'.format(hyperparams['date'],
                                                                                                 hyperparams['hour'],
                                                                                                 hyperparams['batch_size'],
                                                                                                 hyperparams['patch_size'],
                                                                                                 hyperparams['learning_rate'],
                                                                                                 hyperparams['conv_size']),
                                          sess.graph)
        test_writer = tf.summary.FileWriter('./logs/{}/hour{}/test/bs{}_ps{}_lr{}_cs{}'.format(hyperparams['date'],
                                                                                                 hyperparams['hour'],
                                                                                                 hyperparams['batch_size'],
                                                                                                 hyperparams['patch_size'],
                                                                                                 hyperparams['learning_rate'],
                                                                                                 hyperparams['conv_size']),
                                            sess.graph)

        for ep in tqdm(range(hyperparams['nb_epoch']), desc='Epoch'):  # fixme: tqdm print new line after an exception
            sess.run(train_inputs['iterator_init_op'], feed_dict={train_inputs['fnames_ph']: hyperparams['totrain_files'],
                                                                  # train_inputs['patch_size_ph']: str(hyperparams['patch_size'])})
                                                                  train_inputs['patch_size_ph']: [hyperparams['patch_size']] * len(hyperparams['totrain_files'])})
            sess.run(test_inputs['iterator_init_op'], feed_dict={test_inputs['fnames_ph']: hyperparams['totest_files'],
                                                                 # test_inputs['patch_size_ph']: str(hyperparams['patch_size'])})
                                                                 test_inputs['patch_size_ph']: [hyperparams['patch_size']] * len(hyperparams['totest_files'])})
            # begin training
            for step in tqdm(range(hyperparams['nb_batch']), desc='Batch step'):
                try:
                    # 80%train 10%cross-validation 10%test
                    if step % 9 == 8:
                        # 5 percent of the data will be use to cross-validation
                        summary, _ = sess.run([nodes['summary'], nodes['train_or_test_op']],
                                              feed_dict={nodes['train_or_test']: 'cv',
                                                         nodes['drop']: 1})
                        cv_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)

                        # in situ testing without loading weights like cs-230-stanford
                        summary, _ = sess.run([nodes['summary'], nodes['train_or_test_op']],
                                              feed_dict={nodes['train_or_test']: 'test',
                                                         nodes['drop']: 1})
                        test_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)

                    # 90 percent of the data will be use for training
                    else:
                        summary, _ = sess.run([nodes['summary'], nodes['train_or_test_op']],
                                              feed_dict={nodes['train_or_test']: 'train',
                                                         nodes['drop']: 0.5})
                        train_writer.add_summary(summary, ep * hyperparams['nb_batch'] + step)

                except tf.errors.OutOfRangeError as e:
                    print(e)
                    break

                if step % save_step == 0:
                    # prepare input dict and out dict
                    in_dict = {
                        'train_files_ph': train_inputs['fnames_ph'],
                        'train_ps_ph': train_inputs['patch_size_ph'],
                        'test_files_ph': test_inputs['fnames_ph'],
                        'test_ps_ph': test_inputs['patch_size_ph'],
                    }
                    out_dict = {
                        'prediction': nodes['y_pred'],
                        'tot_op': nodes['train_or_test_op'],
                        'img': nodes['img'],
                        'label': nodes['label']
                    }
                    # builder
                    tf.saved_model.simple_save(sess, './logs/{}/hour{}/savedmodel/step{}/'.format(hyperparams['date'],
                                                                                                  hyperparams['hour'],
                                                                                                  step + ep * hyperparams['nb_batch']),
                                               in_dict, out_dict)

