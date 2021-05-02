# class MBGDHelper:
#     '''Mini Batch Grandient Descen helper'''
#     def __init__(self, batch_size, patch_size):
#         self.i = 0
#         self.batch_size = batch_size
#         self.patch_size = patch_size
#         self.epoch_len = self._epoch_len()
#         self.order = np.arange(self.epoch_len) #data has been pre-shuffle
#         self.onoff = False
#     def next_batch(self):
#         try:
#             try:
#                 with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
#                     tmp = self.order.tolist()[self.i * self.batch_size: (self.i + 1) * self.batch_size]
#                     X = f['X'][sorted(tmp)].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
#                     y = f['y'][sorted(tmp)].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
#                     idx = np.random.permutation(X.shape[0])
#                 self.i += 1
#                 return X[idx], y[idx]
#             except:
#                 print('\n***Load last batch')
#                 with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
#                     modulo = f['X'].shape % self.batch_size
#                     tmp = self.order.tolist()[-modulo:]
#                     X = f['X'][sorted(tmp)].reshape(modulo, self.patch_size, self.patch_size, 1)
#                     y = f['y'][sorted(tmp)].reshape(modulo, self.patch_size, self.patch_size, 1)
#                     idx = np.random.permutation(X.shape[0])
#                 self.i += 1
#                 return X[idx], y[idx]
#         except Exception as ex:
#             raise ex
#
#     def _epoch_len(self):
#         with h5py.File('./proc/{}.h5'.format(self.patch_size), 'r') as f:
#             print('Total epoch number is {}'.format(f['X'].shape[0]))
#             return f['X'].shape[0]
#
#     def get_epoch(self):
#         return self.epoch_len
#
#     def shuffle(self):
#         np.random.shuffle(self.order)
#         self.i = 0
#         print('shuffled datas')
#
#
# class MBGD_Helper_v2(object):
#     def __init__(self,
#                  batch_size,
#                  patch_size,
#                  coord,
#                  max_queue_size=32
#                  ):
#
#         # init params
#         self.batch_size = batch_size
#         self.patch_size = patch_size
#         self.flist = self._init_flist()
#         self.flist_len = len(self.flist)
#
#         # init fifo queue
#         self.max_queue_size = max_queue_size
#         self.queue = tf.PaddingFIFOQueue(max_queue_size, ['float32'], shapes=[(None, None)])
#         self.queue_size = self.queue.size()
#         self.threads = []
#         self.coord = coord
#         self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
#         self.enqueue = self.queue.enqueue([self.sample_placeholder])
#         self.i = 0
#         self.onoff = False
#
#     def _init_flist(self):
#         flist = []
#         for dirpath, _, fnames in os.walk('./proc/'):
#             for fname in fnames:
#                 if fname.startswith('{}_{}'.format(self.patch_size, self.batch_size)):
#                     flist.append(fname)
#         return flist
#
#     def load_data(self):
#         print('thread id: {}'.format(threading.get_ident()))
#         with h5py.File('./proc/{}_{}_.h5'.format(self.patch_size, self.batch_size), 'r') as f:
#             X = f['X'].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
#             y = f['y'].reshape(self.batch_size, self.patch_size, self.patch_size, 1)
#             idx = np.random.permutation(X.shape[0])
#         yield X[idx], y[idx]
#
#     def dequeue(self, nb_batch=1):
#         output = self.queue.dequeue_many(nb_batch)
#         return output
#
#     def thread_main(self, sess):
#         stop = False
#         while not stop:
#             iterator = self.load_data()
#             for data in iterator:
#                 while self.queue_size.eval(session=sess) == self.max_queue_size:
#                     if self.coord.should_stop():
#                         break
#
#                 if self.coord.should_stop():
#                     stop = True
#                     print("Enqueue thread receives stop request.")
#                     break
#                 sess.run(self.enqueue, feed_dict={self.sample_placeholder: data})
#
#     def start_threads(self, sess, n_threads=mp.cpu_count()):
#         for _ in range(n_threads):
#             thread = threading.Thread(target=self.thread_main, args=(sess,))
#             thread.daemon = True  # Thread will close when parent quits.
#             thread.start()
#             self.threads.append(thread)
#         return self.threads
#
#
# class MBGD_Helper_v3:
#     def __init__(self, patch_size, batch_size):
#         self.patch_size = patch_size
#         self.batch_size = batch_size
#         self.X_flist, self.y_flist = self._init_flist()
#         self.len_flist = len(self.X_flist)
#
#     def fetch(self, X_fname, y_fname):
#         record_defaults = [[1], [1]*self.patch_size*self.patch_size*self.batch_size]
#         X = tf.read_file(X_fname)
#         y = tf.read_file(y_fname)
#         X = tf.decode_csv(X, record_defaults=record_defaults, field_delim=',')
#         y = tf.decode_csv(y, record_defaults=record_defaults, field_delim=',')
#         X = tf.reshape(X, [self.batch_size, self.patch_size, self.patch_size, 1])
#         y = tf.reshape(y, [self.batch_size, self.patch_size, self.patch_size, 1])
#         return X, y
#
#     def _init_flist(self):
#         X_flist = []
#         y_flist = []
#         for dirpath, _, fnames in os.walk('./proc/'):
#             for fname in fnames:
#                 if fname.startswith('X{}_{}'.format(self.patch_size, self.batch_size)) and \
#                         fname.endswith('csv'):
#                     X_flist.append(fname)
#                 elif fname.startswith('y{}_{}'.format(self.patch_size, self.batch_size)) and \
#                         fname.endswith('csv'):
#                     y_flist.append(fname)
#         return X_flist, y_flist
#
#     def load_data(self):
#         dataset = tf.data.Dataset.from_tensor_slices((self.X_flist, self.y_flist))
#         dataset = dataset.shuffle(self.len_flist)
#         dataset = dataset.map(self.fetch, num_parallel_calls=mp.cpu_count())
#         dataset = dataset.batch(1)
#         dataset = dataset.prefetch(3)
#         X, y = dataset.make_one_shot_iterator().get_next()
#         return X, y
#         # return dataset
#
#
# class MBGD_Helper_v4:
#     def __call__(self, fname, patch_size, batch_size, io):
#         with h5py.File(fname, 'r') as f:
#             if io == 'X':
#                 X = f['X'].reshape(batch_size, patch_size, patch_size, 1)
#                 yield X
#             else:
#                 y = f['y'].reshape(batch_size, patch_size, patch_size, 1)
#                 yield y
#
#
# def MBGDHelper_v5(patch_size, batch_size, ncores=mp.cpu_count()):
#     '''
#     tensorflow tf.data input pipeline based helper that return batches of images and labels at once
#
#     input:
#     -------
#     patch_size: (int) pixel length of one small sampling window (patch)
#     batch_size: (int) number of images per batch before update parameters
#
#     output:
#     -------
#     inputs: (dict) output of this func, but inputs of the neural network. A dictionary of batch and the iterator
#     initialization operation
#     '''
#     # init list of files
#     files = tf.data.Dataset.list_files('./proc/{}_{}_*.h5'.format(patch_size, batch_size))
#     dataset = files.map(_pyfn_wrapper, num_parallel_calls=ncores)
#     dataset = dataset.batch(1).prefetch(ncores + 1)  #batch() should be 1 here because 1 .h5 file for 1 batch
#
#     # construct iterator
#     it = dataset.make_initializable_iterator()
#     iter_init_op = it.initializer
#
#     # get next batch
#     X_it, y_it = it.get_next()
#     inputs = {'imgs': X_it, 'labels': y_it, 'iterator_init_op': iter_init_op}
#     return inputs
#
#
# def parse_h5(name, patch_size=40, batch_size=1000):
#     '''
#     parser that return the input images and  output labels
#
#     input:
#     -------
#     name: (bytes literal) file name
#
#     output:
#     -------
#     X: (numpy ndarray) reshape array as dataformat 'NHWC'
#     y: (numpy ndarray) reshape array as dataformat 'NHWC'
#     '''
#     with h5py.File(name.decode('utf-8'), 'r') as f:
#         X = f['X'][:].reshape(batch_size, patch_size, patch_size, 1)
#         y = f['y'][:].reshape(batch_size, patch_size, patch_size, 1)
#         return _minmaxscalar(X), _minmaxscalar(y)
#
#
# def _pyfn_wrapper(filename, patch_size, batch_size):
#     '''
#     input:
#     -------
#     filename: (tf.data.Dataset)  Tensors of strings
#
#     output:
#     -------
#     function: (function) tensorflow's pythonic function with its arguements
#     '''
#     return tf.py_func(parse_h5,  #wrapped pythonic function
#                       [filename, patch_size, batch_size],
#                       [tf.float32, tf.int8]  #[input, output] dtype #fixme: maybe gpu version doesn't have algorithm for int8
#
#
# ################################################ Neurons layers#########################################################
# # input_size = 784
# # hidden_layer_neurons = 10
# # output_size = 784
# # learning_rate = 0.001
# # epoch = 1000
# # batch_size = 5
# #
# # X = tf.placeholder(tf.float32, [None, input_size], name="input_X")
# # y = tf.placeholder(tf.float32, [None, output_size], name="Output_y")
# # X_img = tf.reshape(X, [-1, 28, 28, 1])
# # y_img = tf.reshape(X, [-1, 28, 28, 1])
# # tf.summary.image('X_img', X_img, 1)
# # tf.summary.image('y_img', y_img, 1)
# #
# # # First layer of weights
# # with tf.name_scope("layer1"):
# #     W1 = tf.get_variable("W1", shape=[input_size, hidden_layer_neurons],
# #                          initializer=tf.contrib.layers.xavier_initializer())
# #     layer1 = tf.matmul(X, W1)
# #     layer1_act = tf.nn.tanh(layer1)
# #     tf.summary.histogram("weights", W1)
# #     tf.summary.histogram("layer", layer1)
# #     tf.summary.histogram("activations", layer1_act)
# #
# # # Second layer of weights
# # with tf.name_scope("layer2"):
# #     W2 = tf.get_variable("W2", shape=[hidden_layer_neurons, hidden_layer_neurons],
# #                          initializer=tf.contrib.layers.xavier_initializer())
# #     layer2 = tf.matmul(layer1_act, W2)
# #     layer2_act = tf.nn.tanh(layer2)
# #     tf.summary.histogram("weights", W2)
# #     tf.summary.histogram("layer", layer2)
# #     tf.summary.histogram("activations", layer2_act)
# #
# # # Third layer of weights
# # with tf.name_scope("layer3"):
# #     W3 = tf.get_variable("W3", shape=[hidden_layer_neurons, hidden_layer_neurons],
# #                          initializer=tf.contrib.layers.xavier_initializer())
# #     layer3 = tf.matmul(layer2_act, W3)
# #     layer3_act = tf.nn.tanh(layer3)
# #
# #     tf.summary.histogram("weights", W3)
# #     tf.summary.histogram("layer", layer3)
# #     tf.summary.histogram("activations", layer3_act)
# #
# # # Fourth layer of weights
# # with tf.name_scope("layer4"):
# #     W4 = tf.get_variable("W4", shape=[hidden_layer_neurons, output_size],
# #                          initializer=tf.contrib.layers.xavier_initializer())
# #     Qpred = tf.nn.softmax(tf.matmul(layer3_act, W4)) # Bug fixed: Qpred = tf.nn.softmax(tf.matmul(layer3, W4))
# #     tf.summary.histogram("weights", W4)
# #     tf.summary.histogram("Qpred", Qpred)# First layer of weights
#
#
# ##############################################Convolution layer ########################################################
# # input_size = 784
# # hidden_layer_neurons = 10
# # output_size = 784
# # learning_rate = 0.001
# # epoch = 1000
# # batch_size = 5
# #
# # X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input_X")
# # y = tf.placeholder(tf.float32, [None, 14, 14, 1], name="Output_y")
# # X_img = tf.reshape(X, [-1, 28, 28, 1])
# # y_img = tf.reshape(X, [-1, 28, 28, 1])
# # tf.summary.image('X_img', X_img, 1)
# # tf.summary.image('y_img', y_img, 1)
# #
# # # C1
# # with tf.name_scope("layer1"):
# #     W1 = tf.get_variable("W1", shape=[3, 3, 1, 32],
# #                          initializer=tf.contrib.layers.xavier_initializer())
# #     b1 = tf.get_variable("b1", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
# #     layer1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
# #     layer1_act = tf.nn.relu(layer1)
# #     tf.summary.histogram("weights", W1)
# #     tf.summary.histogram("layer", layer1)
# #     tf.summary.histogram("activations", layer1_act)
# #
# # # C2
# # with tf.name_scope("layer2"):
# #     W2 = tf.get_variable("W2", shape=[3, 3, 32, 64],
# #                          initializer=tf.contrib.layers.xavier_initializer())
# #     b2 = tf.get_variable("b2", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
# #     layer2 = tf.nn.conv2d(layer1_act, W2, strides=[1, 1, 1, 1], padding='SAME') + b2
# #     layer2_act = tf.nn.relu(layer2)
# #     tf.summary.histogram("weights", W2)
# #     tf.summary.histogram("layer", layer2)
# #     tf.summary.histogram("activations", layer2_act)
# #
# # # max pool
# # with tf.name_scope("maxpool"):
# #     maxpool = tf.nn.max_pool(layer2_act, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
# #
# # # C3
# # with tf.name_scope("layer3"):
# #     W3 = tf.get_variable("W3", shape=[3, 3, 64, 32],
# #                          initializer=tf.contrib.layers.xavier_initializer())
# #     b3 = tf.get_variable("b3", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
# #     layer3 = tf.nn.conv2d(maxpool, W3, strides=[1, 1, 1, 1], padding='SAME') + b3
# #     layer3_act = tf.nn.relu(layer3)
# #
# #     tf.summary.histogram("weights", W3)
# #     tf.summary.histogram("layer", layer3)
# #     tf.summary.histogram("activations", layer3_act)
# #
# # # C4
# # with tf.name_scope("layer4"):
# #     W4 = tf.get_variable("W4", shape=[3, 3, 32, 1],
# #                          initializer=tf.contrib.layers.xavier_initializer())
# #     b4 = tf.get_variable("b4", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
# #     Qpred = tf.nn.conv2d(layer3_act, W4, strides=[1, 1, 1, 1], padding='SAME') + b4
# #     tf.summary.histogram("weights", W4)
# #     tf.summary.histogram("Qpred", Qpred)
# #
# # # Loss function
# # with tf.name_scope("loss"):
# #     loss = tf.reduce_mean(tf.losses.mean_squared_error(
# #         labels=tf.cast(y, tf.int32),
# #         predictions=Qpred))
# #     tf.summary.scalar("Q", tf.reduce_mean(Qpred))
# #     tf.summary.scalar("Y", tf.reduce_mean(y))
# #     tf.summary.scalar("loss", loss)
# #
# # # Learning
# # # with tf.name_scope("performance"):
# # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# # print('number of params: {}'.format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))
# # grads = optimizer.compute_gradients(loss)
# # summ_grad = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
# # train_op = optimizer.minimize(loss)
# #
# #
# # merged = tf.summary.merge_all()
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     writer = tf.summary.FileWriter('./dum_logs/', sess.graph, 3)
# #     for i in range(epoch // batch_size):
# #         print(i)
# #         X_batch = np.random.rand(784 * 5).reshape(5, 28, 28, 1)
# #         y_batch = np.random.rand(784 // 4 * 5).reshape(5, 14, 14, 1)
# #         sum, _, grad_vals = sess.run([merged, train_op, summ_grad], feed_dict={X: X_batch, y: y_batch})
# #         writer.add_summary(sum, i)
#
# ############################################multi thread .h5 reader#####################################################
#
# ############################################ FIFO ######################################################################
# #https://github.com/philipperemy/tensorflow-fifo-queue-example/blob/master/main.py
# # import time
# # import threading
# #
# # def load_data():
# #     # yield batches
# #     for i in range(10000):
# #         yield np.random.uniform(size=(5, 5))
# #
# #
# # class DataGenerator(object):
# #     def __init__(self,
# #                  coord,
# #                  max_queue_size=32,
# #                  wait_time=0.01):
# #         # Change the shape of the input data here with the parameter shapes.
# #         self.wait_time = wait_time
# #         self.max_queue_size = max_queue_size
# #         self.queue = tf.PaddingFIFOQueue(max_queue_size, ['float32'], shapes=[(None, None)])
# #         self.queue_size = self.queue.size()
# #         self.threads = []
# #         self.coord = coord
# #         self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)
# #         self.enqueue = self.queue.enqueue([self.sample_placeholder])
# #
# #     def dequeue(self, num_elements):
# #         output = self.queue.dequeue_many(num_elements)
# #         return output
# #
# #     def thread_main(self, sess):
# #         stop = False
# #         while not stop:
# #             iterator = load_data()
# #             for data in iterator:
# #                 while self.queue_size.eval(session=sess) == self.max_queue_size:
# #                     if self.coord.should_stop():
# #                         break
# #                     time.sleep(self.wait_time)
# #                 if self.coord.should_stop():
# #                     stop = True
# #                     print("Enqueue thread receives stop request.")
# #                     break
# #                 sess.run(self.enqueue, feed_dict={self.sample_placeholder: data})
# #
# #     def start_threads(self, sess, n_threads=16):
# #         for _ in range(n_threads):
# #             thread = threading.Thread(target=self.thread_main, args=(sess,))
# #             thread.daemon = True  # Thread will close when parent quits.
# #             thread.start()
# #             self.threads.append(thread)
# #         return self.threads
# #
# # def define_net(input_batch):
# #     return input_batch + 20  # simplest network I could think of.
# #
# #
# # def main():
# #     batch_size = 5
# #
# #     coord = tf.train.Coordinator()
# #     with tf.name_scope('create_inputs'):
# #         reader = DataGenerator(coord)
# #         input_batch = reader.dequeue(batch_size)
# #
# #     gpu_options = tf.GPUOptions(visible_device_list='0')
# #     sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,
# #                                             log_device_placement=True))
# #     init = tf.global_variables_initializer()
# #     sess.run(init)
# #
# #     threads = reader.start_threads(sess)
# #     net = define_net(input_batch)
# #     queue_size = reader.queue_size
# #
# #     for step in range(10000):
# #         # run_meta = tf.RunMetadata()
# #         print('size queue =', queue_size.eval(session=sess))
# #         _ = sess.run(net, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
# #         print(_)
# #         # profiler.add_step(step, run_meta=run_meta)
# #
# #         # Profile the params
# #         # profiler.profile_name_scope(options=(option_builder.ProfileOptionBuilder.trainable_variables_parameter()))
# #         # opts = option_builder.ProfileOptionBuilder.time_and_memory()
# #
# #         # Make this thread slow. You can comment this line. If you do so, you will dequeue
# #         # faster than you enqueue, so expect the queue not to reach its maximum (32 by default)
# #         # time.sleep(1)
# #
# #     coord.request_stop()
# #     print("stop requested.")
# #     for thread in threads:
# #         thread.join()
# #
# # main()
#
# ########################################### FIFO #######################################################################
# # https://github.com/adventuresinML/adventures-in-ml-code/blob/master/tf_queuing.py
# # import h5py
# # import threading
# # import os
# #
# # batch_size = 1000
# # num_threads = 16
# # patch_size = 40
# #
# # def read_data(file_q):
# #     print('thread id: {}'.format(threading.get_ident()))
# #     inputs, _ = file_q.dequeue_many(1)
# #     # with h5py.File('./proc/{}_{}.h5'.format(patch_size, batch_size), 'r') as f:
# #     with h5py.File(inputs, 'r') as f:
# #         X = f['X'].reshape(batch_size, patch_size, patch_size, 1)
# #         y = f['y'].reshape(batch_size, patch_size, patch_size, 1)
# #         return X, y
# #
# # def cifar_shuffle_batch():
# #     flist = []
# #     for dirpath, _, fnames in os.walk('./proc/'):
# #         for fname in fnames:
# #             if fname.startswith('{}_{}'.format(patch_size, batch_size)):
# #                 flist.append(fname)
# #
# #     file_q = tf.train.string_input_producer(flist)
# #     image, label = read_data(file_q)
# #     min_after_dequeue = 10000
# #     capacity = min_after_dequeue + (num_threads + 1) * batch_size
# #     image_batch, label_batch = cifar_shuffle_queue_batch(image,
# #                                                          label,
# #                                                          batch_size,
# #                                                          capacity,
# #                                                          min_after_dequeue,
# #                                                          num_threads,
# #                                                          )
# #
# #     # now run the training
# #     cifar_run(image_batch, label_batch)
# #
# # def cifar_run(image, label):
# #     with tf.Session() as sess:
# #         coord = tf.train.Coordinator()
# #         threads = tf.train.start_queue_runners(coord=coord)
# #         for i in range(5):
# #             image_batch, label_batch = sess.run([image, label])
# #             print(image_batch.shape, label_batch.shape)
# #
# #         coord.request_stop()
# #         coord.join(threads)
# #
# # def cifar_filename_queue(filename_list):
# #     # convert the list to a tensor
# #     string_tensor = tf.convert_to_tensor(filename_list, dtype=tf.string)
# #     # randomize the tensor
# #     tf.random_shuffle(string_tensor)
# #     # create the queue
# #     fq = tf.FIFOQueue(capacity=10, dtypes=tf.string)
# #     # create our enqueue_op for this q
# #     fq_enqueue_op = fq.enqueue_many([string_tensor])
# #     # create a QueueRunner and add to queue runner list
# #     # we only need one thread for this simple queue
# #     tf.train.add_queue_runner(tf.train.QueueRunner(fq, [fq_enqueue_op] * 1))
# #     return fq
# #
# # def cifar_shuffle_queue_batch(image, label, batch_size, capacity, min_after_dequeue, threads):
# #     tensor_list = [image, label]
# #     dtypes = [tf.float32, tf.int32]
# #     shapes = [image.get_shape(), label.get_shape()]
# #     q = tf.RandomShuffleQueue(capacity=capacity, min_after_dequeue=min_after_dequeue,
# #                               dtypes=dtypes, shapes=shapes)
# #     enqueue_op = q.enqueue(tensor_list)
# #     # add to the queue runner
# #     tf.train.add_queue_runner(tf.train.QueueRunner(q, [enqueue_op] * threads))
# #     # now extract the batch
# #     image_batch, label_batch = q.dequeue_many(batch_size)
# #     return image_batch, label_batch
# #
# #
# # if __name__ == "__main__":
# #     run_opt = 3
# #     if run_opt == 1:
# #         pass
# #     elif run_opt == 2:
# #         pass
# #     elif run_opt == 3:
# #         cifar_shuffle_batch()
#
# ##################################### tf.data interleave################################################################
# # import h5py
# # import os
# # import multiprocessing as mp
# ################# https://stackoverflow.com/questions/50046505/how-to-use-parallel-interleave-in-tensorflow#############
# # class generator:
# #     def __call__(self, path, io):
# #         with h5py.File(path, 'r') as f:
# #             if io == 'X':
# #                 X = f['X'].reshape(batch_size, patch_size, patch_size, 1)
# #                 return X
# #             else:
# #                 y = f['y'].reshape(batch_size, patch_size, patch_size, 1)
# #                 return y
# #
# # def generator_returnX(path):
# #     with h5py.File(path, 'r') as f:
# #         return f['X'][:]
# #
# # def generator_returny(path):
# #     with h5py.File(path, 'r') as f:
# #         return f['y'][:]
# #
# # flist = []
# # for dirpath, _, fnames in os.walk('./proc/'):
# #     for fname in fnames:
# #         if fname.startswith('{}_{}'.format(patch_size, batch_size)) and fname.endswith('h5'):
# #             flist.append(fname)
# #
# # ds = tf.data.Dataset.from_tensor_slices(flist)
# # X = ds.apply(tf.data.experimental.parallel_interleave(lambda filename: tf.data.Dataset.from_generator(
# #     generator_returnX, output_types=tf.float32, output_shapes=tf.TensorShape([10000, 40, 40])),
# #                                                               cycle_length=len(flist), sloppy=False))
# # y = ds.apply(tf.data.experimental.parallel_interleave(lambda filename: tf.data.Dataset.from_generator(
# #     generator_returny, output_types=tf.float32, output_shapes=tf.TensorShape([10000, 40, 40])),
# #                                                               cycle_length=len(flist), sloppy=False))
# # print(X, y)
# # X = X.cache()
# # y = y.cache()
# # X_it = X.make_one_shot_iterator()
# # y_it = y.make_one_shot_iterator()
# #########https://stackoverflow.com/questions/50046505/how-to-use-parallel-interleave-in-tensorflow######################
# # X, y = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
# #         generator(),
# #         tf.uint8,
# #         tf.TensorShape([None, patch_size, patch_size, 1]),
# #         args=(filename,)),
# #        cycle_length=4, block_length=4)
# # print(X, y)
# # y = tf.data.Dataset.from_tensor_slices((flist, 'y'))
# # y = y.interleave(lambda filename: tf.data.Dataset.from_generator(
# #         generator(),
# #         tf.uint8,
# #         tf.TensorShape([None, patch_size, patch_size, 1]),
# #         args=(filename,)),
# #        cycle_length=4, block_length=4)
# # print(X, y)
# # # y = tf.data.Dataset.from_tensor_slices((flist, 'y'))
# # # y = y.interleave(lambda filename: tf.data.Dataset.from_generator(
# # #         generator(),
# # #         tf.uint8,
# # #         tf.TensorShape([None,patch_size,patch_size,1]),
# # #         args=(filename,)),
# # #        cycle_length=1, block_length=1)
# #
# # # def load_data(path):
# # #     with h5py.File(path, 'r') as f:
# # #         X = f['X'].reshape(batch_size, patch_size, patch_size, 1)
# # #         y = f['y'].reshape(batch_size, patch_size, patch_size, 1)
# # #         return X, y
# # # X, y = flist.map(load_data, num_parallel_calls=mp.cpu_count())\
# # #     .apply(tf.contrib.data.shuffle_and_repeat(100)).batch(1).prefetch(3)
# # #
# # # X_batch = flist.apply(tf.data.experimental.parallel_interleave(
# # #     lambda filename: tf.data.Dataset.from_generator(
# # #         Generator,
# # #         tf.uint8,
# # #         tf.TensorShape([batch_size, patch_size, patch_size, 1]),
# # #         args=(filename, 'X')),
# # #     cycle_length=4,
# # #     block_length=8
# # # )
# # # )
# # # y_batch = fnames.apply(tf.data.experimental.parallel_interleave(
# # #     lambda filename: tf.data.Dataset.from_generator(
# # #         Generator,
# # #         tf.uint8,
# # #         tf.TensorShape([batch_size, patch_size, patch_size, 1]),
# # #         args=(filename, 'y')),
# # #     cycle_length=4,
# # #     block_length=8
# # # )
# # # )
# # #
# # # X_batch = X_batch.cache()
# # # y_batch = y_batch.cache()
# # #
# # # X_img = X_batch.map(read_decode, num_parallel_calls=20)\
# # #     .apply(tf.contrib.data.shuffle_and_repeat(100))\
# # #     .batch(batch_size)\
# # #     .prefetch(5)
# # #
# # # y_img = y_batch.map(read_decode, num_parallel_calls=20)\
# # #     .apply(tf.contrib.data.shuffle_and_repeat(100))\
# # #     .batch(batch_size)\
# # #     .prefetch(5)
# #
# # # model
# # X_ph = tf.placeholder(tf.float32, shape=None)
# # y_ph = tf.placeholder(tf.float32, shape=None)
# # W = tf.get_variable('w', shape=[conv_size, conv_size, 1, 1], initializer=tf.contrib.layers.xavier_initializer())
# # loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y_ph, predictions=tf.matmul(X_ph, W)))
# # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# #
# # # session
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     sess.run(train_op, feed_dict={X_ph: X, y_ph: y})
#
# ##################################### tf.data prefetch##################################################################
# # import h5py
# # import os
# #
# # patch_size = 40
# # batch_size = 1000
# # conv_size = 3
# # nb_conv = 32
# # learning_rate = 0.0001
# #
# # # define parser function
# # def parse_function(fname):
# #     with h5py.File(fname, 'r') as f:
# #         X = f['X'].reshape(batch_size, patch_size, patch_size, 1)
# #         y = f['y'].reshape(batch_size, patch_size, patch_size, 1)
# #         return X, y
# #
# # # create a list of files path
# # flist = []
# # for dirpath, _, fnames in os.walk('./proc/'):
# #     for fname in fnames:
# #         if fname.startswith('{}_{}'.format(patch_size, batch_size)) and fname.endswith('h5'):
# #             flist.append(fname)
# #
# # # prefetch data
# # dataset = tf.data.Dataset.from_tensor_slices((tf.constant(flist)))
# # dataset = dataset.shuffle(len(flist))
# # dataset = dataset.map(parse_function, num_parallel_calls=4)
# # dataset = dataset.batch(1)
# # dataset = dataset.prefetch(3)
# # X_it, y_it = dataset.make_initializable_iterator().get_next()
# # # simplest model that I think of
# # W = tf.get_variable('w', shape=[conv_size, conv_size, 1, 1], initializer=tf.contrib.layers.xavier_initializer())
# # loss = tf.reduce_mean(tf.losses.mean_squared_error(tf.nn.softmax(labels=y_it, predictions=tf.matmul(X_it, W))))
# # train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# #
# # # start session
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     print(sess.run(train_op))
#
# ###https://stackoverflow.com/questions/52179857/parallelize-tf-from-generator-using-tf-contrib-data-parallel-interleave#
# #
# import h5py
# import threading
# from tqdm import tqdm
# import multiprocessing as mp
#
# def write_h5(x):
#     with h5py.File('./proc/test_{}.h5'.format(x), 'w') as f:
#             print(mp.current_process())  # see process ID
#             a = np.ones((1000, 100, 100))
#             b = np.dot(a, 3)
#             f.create_dataset('X', shape=(1000, 100, 100), dtype='float32', data=a)
#             f.create_dataset('y', shape=(1000, 100, 100), dtype='float32', data=b)
#
# # p = mp.Pool(mp.cpu_count())
# # p.map(write_h5, range(100))
#
# shuffle_size = prefetch_buffer = 1
# batch_size = 1
#
#
# def parse_file(f):
#     print(f.decode('utf-8'))
#     with h5py.File(f.decode("utf-8"), 'r') as fi:
#         X = fi['X'][:].reshape(100, 100, 1000)
#         y = fi['y'][:].reshape(100, 100, 1000)
#         return X, y
#
#
# def parse_file_tf(filename):
#     return tf.py_func(parse_file, [filename], [tf.float32, tf.float32])
#
# files = tf.data.Dataset.list_files('./proc/test_*.h5')
# dataset = files.map(parse_file_tf, num_parallel_calls=mp.cpu_count())
# dataset = dataset.batch(batch_size).shuffle(shuffle_size).prefetch(5)
# it = dataset.make_initializable_iterator()
# iter_init_op = it.initializer
# X_it, y_it = it.get_next()
#
# # C1
# W1 = tf.get_variable("W1", shape=[3, 3, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
# b1 = tf.get_variable("b1", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
# layer1 = tf.nn.relu(tf.nn.conv2d(X_it, W1, strides=[1, 1, 1, 1], padding='SAME') + b1)
#
# # C2
# W2 = tf.get_variable("W2", shape=[3, 3, 32, 1], initializer=tf.contrib.layers.xavier_initializer())
# b2 = tf.get_variable("b2", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
# layer2 = tf.nn.relu(tf.nn.conv2d(layer1, W2, strides=[1, 1, 1, 1], padding='SAME') + b2)
#
# # MP
# maxpool = tf.nn.max_pool(layer2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
#
# # UP
# up = tf.image.resize_nearest_neighbor(maxpool, [100, 100])
#
# # D3
# W3 = tf.get_variable("W3", shape=[3, 3, 64, 1], initializer=tf.contrib.layers.xavier_initializer())
# b3 = tf.get_variable("b3", shape=[64], initializer=tf.contrib.layers.xavier_initializer())
# layer3 = tf.nn.conv2d_transpose(up, W3, output_shape=(batch_size,
#                                                       int(up.shape[1]),
#                                                       int(up.shape[2]),
#                                                       int(W3.shape[2])),
#                                 strides=[1, 1, 1, 1], padding='SAME') + b3
#
# # D4
# W4 = tf.get_variable("W4", shape=[3, 3, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
# b4 = tf.get_variable("b4", shape=[32], initializer=tf.contrib.layers.xavier_initializer())
# Qpred = tf.nn.conv2d_transpose(layer3, W4, output_shape=(batch_size,
#                                                          int(layer3.shape[1]),
#                                                          int(layer3.shape[2]),
#                                                          int(W4.shape[2])),
#                                strides=[1, 1, 1, 1], padding='SAME') + b4
#
# # Loss function
# loss = tf.reduce_mean(tf.losses.mean_squared_error(
#         labels=tf.cast(y_it, tf.int32),
#         predictions=Qpred))
#
#
# # Train_op
# opt = tf.train.AdamOptimizer(0.0001)
# grads = opt.compute_gradients(loss)
# train_op = opt.minimize(loss)
# m = tf.summary.merge([tf.summary.histogram('w1', W1),
#                       tf.summary.histogram('b1', b1),
#                       tf.summary.histogram('W2', W2),
#                       tf.summary.histogram('b2', b2),
#                       tf.summary.histogram('W3', W3),
#                       tf.summary.histogram('b3', b3),
#                       tf.summary.histogram('W4', W4),
#                       tf.summary.histogram('b4', b4),
#                       tf.summary.scalar("loss", loss),
#                       [tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads]
#                       ])
#
# # session
# sess = tf.Session()
# writer = tf.summary.FileWriter('./dummy', sess.graph)
# sess.run(tf.global_variables_initializer())
# sess.run(iter_init_op)
# for i in tqdm(range(30)):
#     sess.run([train_op])
#     writer.add_summary(m.eval(session=sess), i)
# sess.close()

#########################################nested arguments before pass to map func#######################################
# import tensorflow as tf
# import numpy as np
# import os
# import h5py
# import multiprocessing as mp
#
#
# def write_h5(x):
#     with h5py.File('./proc/test_{}.h5'.format(x), 'w') as f:
#             print(mp.current_process())  # see process ID
#             x = y = np.arange(-1, 1, 0.02)
#             xx, _ = np.meshgrid(x, y)
#             a = xx ** 2
#             b = np.add(a, np.random.randn(100, 100))  #do something and add gaussian noise
#             f.create_dataset('X', shape=(100, 100), dtype='float32', data=a)
#             f.create_dataset('y', shape=(100, 100), dtype='float32', data=b)
#
#
#
# def helper(window_size, batch_size, ncores=mp.cpu_count()):
#     flist = []
#     for dirpath, _, fnames in os.walk('./proc/'):
#         for fname in fnames:
#             if fname.startswith('test') and fname.endswith('.h5'):
#                 flist.append((os.path.abspath(os.path.join(dirpath, fname)), str(window_size)))
#     f_len = len(flist)
#     print(f_len)
#
#     # init list of files
#     batch = tf.data.Dataset.from_tensor_slices((tf.constant(flist)))  #fixme: how to zip one list of string and a list of int
#     batch = batch.map(_pyfn_wrapper, num_parallel_calls=ncores)  #fixme: how to map two args
#     batch = batch.batch(batch_size, drop_remainder=True).prefetch(ncores + 6).shuffle(batch_size)
#
#     # construct iterator
#     it = batch.make_initializable_iterator()
#     iter_init_op = it.initializer
#
#     # get next img and label
#     X_it, y_it = it.get_next()
#     inputs = {'img': X_it, 'label': y_it, 'iterator_init_op': iter_init_op}
#     return inputs, f_len
#
#
# def _pyfn_wrapper(args):  #fixme: args
#     # filename, window_size = args  #fixme: try to separate args
#     # window_size = 100
#     return tf.py_func(parse_h5,  #wrapped pythonic function
#                       [args],
#                       [tf.float32, tf.float32]  #[input, output] dtype
#                       )
#
# def parse_h5(args):
#     name, window_size = args
#     window_size = int(window_size.decode('utf-8'))
#     name = name.decode('utf-8')
#     with h5py.File(name, 'r') as f:
#         X = f['X'][:].reshape(window_size, window_size, 1)
#         y = f['y'][:].reshape(window_size, window_size, 1)
#         return X, y
#
#
# # init data
# # p = mp.Pool(mp.cpu_count())
# # p.map(write_h5, range(100))
# # create tf.data.Dataset
# helper, f_len = helper(100, 5)
# # inject into model
# with tf.name_scope("Conv1"):
#     W = tf.get_variable("W", shape=[3, 3, 1, 1],
#                          initializer=tf.contrib.layers.xavier_initializer())
#     b = tf.get_variable("b", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
#     layer1 = tf.nn.conv2d(helper['img'], W, strides=[1, 1, 1, 1], padding='SAME') + b
#     logits = tf.nn.relu(layer1)
#
#
# loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=helper['label'], predictions=logits))
# train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
#
# # session
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for ep in range(5):
#         print('ep:{}'.format(ep))
#         sess.run(helper['iterator_init_op'])
#         while True:
#             try:
#                 sess.run([train_op])
#             except tf.errors.OutOfRangeError:
#                 break

###############################double tf.data input pipeline with tf.cond###############################################
# import tensorflow as tf
# import numpy as np
# import os
# import h5py
# import multiprocessing as mp
# from itertools import repeat
# #
# # def write_h5(args):
# #     x, is_training = args
# #     with h5py.File('./{}_{}.h5'.format('train' if is_training else 'test', x), 'w') as f:
# #         h = w = np.arange(-1, 1, 0.02)
# #         hh, _ = np.meshgrid(h, w)
# #         a = hh ** 2
# #         b = np.add(a + 1, np.random.randn(100, 100))  #do something and add gaussian noise
# #         f.create_dataset('X', shape=(100, 100), dtype='float32', data=a)
# #         f.create_dataset('y', shape=(100, 100), dtype='float32', data=b)
# #
# #
# # def input_pipeline(window_size, batch_size, is_train=True, ncores=mp.cpu_count()):
# #     flist = []
# #     for dirpath, _, fnames in os.walk('./'):
# #         for fname in fnames:
# #             if fname.startswith('train' if is_train else 'test') and fname.endswith('.h5'):
# #                 print(fname)
# #                 flist.append((os.path.abspath(os.path.join(dirpath, fname)), str(window_size)))
# #     f_len = len(flist)
# #     print(f_len)
# #     # init list of files
# #     batch = tf.data.Dataset.from_tensor_slices((tf.constant(flist)))
# #     batch = batch.map(_pyfn_wrapper, num_parallel_calls=ncores)
# #     batch = batch.batch(batch_size, drop_remainder=True).prefetch(ncores + 6).shuffle(batch_size).repeat()
# #
# #     # construct iterator
# #     it = batch.make_initializable_iterator()
# #     iter_init_op = it.initializer
# #
# #     # get next img and label
# #     X_it, y_it = it.get_next()
# #     inputs = {'img': X_it, 'label': y_it, 'iterator_init_op': iter_init_op}
# #     return inputs, f_len
# #
# #
# # def _pyfn_wrapper(args):
# #     return tf.py_func(parse_h5,  #wrapped pythonic function
# #                       [args],
# #                       [tf.float32, tf.float32]  #[input, output] dtype
# #                       )
# #
# # def parse_h5(args):
# #     name, window_size = args
# #     window_size = int(window_size.decode('utf-8'))
# #     with h5py.File(name, 'r') as f:
# #         X = f['X'][:].reshape(window_size, window_size, 1)
# #         y = f['y'][:].reshape(window_size, window_size, 1)
# #         return X, y
# #
# #
# # # init data
# # # p = mp.Pool(mp.cpu_count())
# # # p.map(write_h5, zip(range(9000), repeat(True)))
# # # p.map(write_h5, zip(range(1000), repeat(False)))
# #
# # # hparam
# # ep_len = 90
# # step_len = 9  # run test_op after 9 steps
# #
# # # create tf.data.Dataset
# # train_input, train_len = input_pipeline(100, 5, is_train=True)
# # test_input, test_len = input_pipeline(100, 5, is_train=False)
# #
# #
# # def model(input, reuse=True):
# #     with tf.variable_scope('model', reuse=reuse):
# #         with tf.name_scope("Conv1"):
# #             W = tf.get_variable("W", shape=[3, 3, 1, 1],
# #                                  initializer=tf.contrib.layers.xavier_initializer())
# #             b = tf.get_variable("b", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
# #             layer1 = tf.nn.conv2d(input['img'], W, strides=[1, 1, 1, 1], padding='SAME') + b
# #             logits = tf.nn.relu(layer1)
# #
# #         loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=input['label'], predictions=logits))
# #         return loss
# #
# # train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(model(train_input, False))
# # test_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(model(test_input, True))
# #
# # # session
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     for ep in range(5):
# #         print('ep:{}'.format(ep))
# #         sess.run(train_input['iterator_init_op'])
# #         sess.run(test_input['iterator_init_op'])
# #         for step in range(ep_len):
# #             print('step:{}\r'.format(step))
# #             try:
# #                 sess.run([train_op])
# #                 if step % step_len == (step_len - 1):
# #                     sess.run([test_op])
# #             except tf.errors.OutOfRangeError:
# #                 raise('drop the remainder')
#
# # def preprocess(dir, stride, patch_size, batch_size, mode='tfrecord', shuffle=True):
# #     # import data
# #     X_stack, y_stack, shapes = _tifReader(dir)
# #     outdir = './proc/'
# #
# #     X_patches = _stride(X_stack[0], stride, patch_size)
# #     y_patches = _stride(y_stack[0], stride, patch_size)
# #
# #     # extract patches
# #     for i in range(1, len(X_stack) - 1):
# #         X_patches = np.vstack((X_patches, _stride(X_stack[i], stride, patch_size)))
# #     for i in range(1, len(y_stack) - 1):
# #         y_patches = np.vstack((y_patches, _stride(y_stack[i], stride, patch_size)))
# #
# #     assert X_patches.shape[0] == y_patches.shape[0], 'numbers of raw image: {} and label image: {} are different'.format(X_patches.shape[0], y_patches.shape[0])
# #
# #     # shuffle
# #     if shuffle:
# #         X_patches, y_patches = _shuffle(X_patches, y_patches)
# #
# #     # handle file id
# #     maxId, rest = _idParser(outdir, batch_size, patch_size)
# #     id_length = (X_patches.shape[0] - rest) // batch_size
# #     if mode == 'h5':
# #         _h5Writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId, mode='h5')
# #     elif mode == 'h5s':
# #         _h5Writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId, mode='h5s')
# #     elif mode == 'csvs':
# #         _h5Writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId, mode='csvs')
# #     elif mode == 'tfrecord':
# #         _h5Writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId, mode='tfrecord')
# ##################################repeat trick######################################
# # import numpy as np
# # import tensorflow as tf
# # train = np.arange(909)
# # test = np.arange(103)
# #
# # train_ds = tf.data.Dataset.from_tensor_slices(train).shuffle(10).batch(10).repeat()
# # test_ds = tf.data.Dataset.from_tensor_slices(test).shuffle(10).batch(10).repeat()
# #
# # train_iterator = train_ds.make_initializable_iterator()
# # test_iterator = test_ds.make_initializable_iterator()
# #
# # with tf.Session() as sess:
# #     sess.run(tf.global_variables_initializer())
# #     sess.run(train_iterator.initializer)
# #     sess.run(test_iterator.initializer)
# #     for i in range(len(train) + 1):
# #         print(sess.run(train_iterator.get_next()))
# #         if i % 9 == 8:
# #             print(sess.run(test_iterator.get_next()))
#
# ################################# save/restore with saved_model API and Dataset input pipeline
# import tensorflow as tf
# import numpy as np
# import os
# import multiprocessing as mp
# from tqdm import tqdm
# import h5py
#
# def parse_h5(args):
#     patch_size = 100
#     with h5py.File(args.decode('utf-8'), 'r') as f:
#         X = f['X'][:].reshape(patch_size, patch_size, 1)
#         y = f['y'][:].reshape(patch_size, patch_size, 1)
#         return _minmaxscalar(X), y  #can't do minmaxscalar for y
#
#
# def _minmaxscalar(ndarray, dtype=np.float32):
#     scaled = np.array((ndarray - np.min(ndarray)) / (np.max(ndarray) - np.min(ndarray)), dtype=dtype)
#     return scaled
#
#
# def _pyfn_wrapper(args):
#     return tf.py_func(parse_h5,  #wrapped pythonic function
#                       [args],
#                       [tf.float32, tf.float32]  #[input, output] dtype
#                       )
#
#
# def input_pipeline(file_names_ph):
#     # create new dataset for predict
#     dataset = tf.data.Dataset.from_tensor_slices(file_names_ph)
#
#     # apply list of file names to the py function wrapper for reading files
#     dataset = dataset.map(_pyfn_wrapper, num_parallel_calls=mp.cpu_count())
#
#     # construct batch size
#     dataset = dataset.batch(1).prefetch(mp.cpu_count())
#
#     # initialize iterator
#     iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
#     iterator_initialize_op = iterator.make_initializer(dataset, name='predict_iter_init_op')
#
#     # get image and labels
#     image_getnext_op, label_getnext_op = iterator.get_next()
#     return {'img_next_op': image_getnext_op, 'label_next_op': label_getnext_op, 'iter_init_op': iterator_initialize_op}
#
#
# def model(in_ds, out_ds):
#
#     with tf.name_scope("Conv1"):
#         W = tf.get_variable("W", shape=[3, 3, 1, 1],
#                              initializer=tf.contrib.layers.xavier_initializer())
#         b = tf.get_variable("b", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
#         layer1 = tf.nn.conv2d(in_ds, W, strides=[1, 1, 1, 1], padding='SAME') + b
#         prediction = tf.nn.relu(layer1, name='prediction')
#
#     with tf.name_scope("Operations"):
#         global_step = tf.Variable(0, name='global_step', trainable=False)
#         loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=out_ds, predictions=prediction), name='loss')
#         train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, name='train_op', global_step=global_step)
#         difference_op = tf.cast(tf.equal(prediction, out_ds), dtype=tf.int32, name='difference')
#
#     return {'global_step': global_step, 'loss': loss, 'train_op': train_op, 'diff_op': difference_op, 'predict_op': prediction}


##############################Training####################################
# # create list of file names: ['test_0.h5', 'test_1.h5', ...]
# totrain_files = [os.path.join('./dummy/', f) for f in os.listdir('./dummy/') if f.endswith('.h5')]
# epoch_length = len(totrain_files)
#
# file_names_ph = tf.placeholder(tf.string, shape=(None), name='file_name_ph')
# in_pipeline = input_pipeline(file_names_ph)
# nodes = model(in_pipeline['img_next_op'], in_pipeline['label_next_op'])
# print([n.name for n in tf.get_default_graph().as_graph_def().node])  # add:  if 'file_name_ph' in n.name to filter names
#
#
# with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
#     sess.run([tf.global_variables_initializer(), in_pipeline['iter_init_op']], feed_dict={file_names_ph: totrain_files})
#     for step in tqdm(range(epoch_length)):
#         # run train_op
#         _ = sess.run(nodes['train_op'])
#         # use saver to save weights
#         if step % epoch_length == epoch_length - 1:
#             in_dict = {
#                 'file_names': file_names_ph,
#             }
#             out_dict = {
#                 'predict': nodes['predict_op'],
#                 'diff_op': nodes['diff_op']
#             }
#             tf.saved_model.simple_save(sess, './dummy/savedmodel', in_dict, out_dict)

##############################Predicting####################################
# # input pipeline for predict
# # create list of file names: ['test_0.h5', 'test_1.h5', ...]
# topredict_files = [os.path.join('./predict/', f) for f in os.listdir('./predict/') if f.endswith('.h5')]
# epoch_length = len(topredict_files)
#
# # save prediction images to /results folder
# if not os.path.exists('./results'):
#     os.makedirs('./results')
#
# # restore
# # set to the default graph
# graph2 = tf.Graph()
# with graph2.as_default():
#     with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
#         tf.saved_model.loader.load(
#             sess,
#             [tf.saved_model.tag_constants.SERVING], './dummy/savedmodel'
#         )
#         # import graph
#         # get operation and so on
#         file_names_ph = graph2.get_tensor_by_name('file_name_ph:0')
#         predict_tensor = graph2.get_tensor_by_name('Conv1/prediction:0')
#         diff_tensor = graph2.get_tensor_by_name('Operations/difference:0')
#         iter_init_op = graph2.get_operation_by_name('predict_iter_init_op')
#
#         sess.run(iter_init_op, feed_dict={file_names_ph: topredict_files})
#         for step in tqdm(range(epoch_length)):
#             predict, difference = sess.run([predict_tensor, diff_tensor])
#             print(predict.shape, difference.shape)
#             with h5py.File('./results/{}.h5'.format(step), 'w') as f:
#                 a = f.create_dataset('prediction', (100, 100), dtype='float32')
#                 a[:] = predict.reshape(100, 100)
#                 b = f.create_dataset('difference', (100, 100), dtype='float32', data=difference)
#                 b[:] = difference.reshape(100, 100)
#
# ##########################20190412 new mechanism
# def parse_h5(name, patch_size):
#     print('name:{}, ps:{}'.format(name, patch_size))
#     with h5py.File(name.decode('utf-8'), 'r') as f:
#         X = f['X'][:].reshape(patch_size, patch_size, 1)
#         y = f['y'][:].reshape(patch_size, patch_size, 1)
#         return _minmaxscalar(X), y  #can't do minmaxscalar for y
#
#
# def _minmaxscalar(ndarray, dtype=np.float32):
#     scaled = np.array((ndarray - np.min(ndarray)) / (np.max(ndarray) - np.min(ndarray)), dtype=dtype)
#     return scaled
#
#
# def _pyfn_wrapper(fname, patchsize):
#     return tf.py_func(parse_h5,  #wrapped pythonic function
#                       [fname, patchsize],
#                       [tf.float32, tf.float32]  #[input, output] dtype
#                       )
#
#
# def input_pipeline(fname_ph, ps_ph):
#     # create new dataset for predict
#     dataset = tf.data.Dataset.from_tensor_slices((fname_ph, ps_ph))
#
#     # apply list of file names to the py function wrapper for reading files
#     dataset = dataset.map(_pyfn_wrapper, num_parallel_calls=mp.cpu_count())
#
#     # construct batch size
#     dataset = dataset.batch(1).prefetch(mp.cpu_count())
#
#     # initialize iterator
#     iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
#     iterator_initialize_op = iterator.make_initializer(dataset, name='predict_iter_init_op')
#
#     # get image and labels
#     image_getnext_op, label_getnext_op = iterator.get_next()
#     return {'img_next_op': image_getnext_op, 'label_next_op': label_getnext_op, 'iter_init_op': iterator_initialize_op}
#
#
# def model(in_ds, out_ds):
#
#     with tf.name_scope("Conv1"):
#         W = tf.get_variable("W", shape=[3, 3, 1, 1],
#                              initializer=tf.contrib.layers.xavier_initializer())
#         b = tf.get_variable("b", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
#         layer1 = tf.nn.conv2d(in_ds, W, strides=[1, 1, 1, 1], padding='SAME') + b
#         prediction = tf.nn.relu(layer1, name='prediction')
#
#     with tf.name_scope("Operations"):
#         global_step = tf.Variable(0, name='global_step', trainable=False)
#         loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=out_ds, predictions=prediction), name='loss')
#         train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, name='train_op', global_step=global_step)
#         difference_op = tf.cast(tf.equal(prediction, out_ds), dtype=tf.int32, name='difference')
#
#     return {'global_step': global_step, 'loss': loss, 'train_op': train_op, 'diff_op': difference_op, 'predict_op': prediction}
#
#
# ##############################Training####################################
# # create list of file names: ['test_0.h5', 'test_1.h5', ...]
# totrain_files = [os.path.join('./dummy/', f) for f in os.listdir('./dummy/') if f.endswith('.h5')]
# epoch_length = len(totrain_files)
# # args = [str((fname, 100)) for fname in totrain_files]
# # print(args)
#
# fname_ph = tf.placeholder(tf.string, shape=(None), name='file_name_ph')
# ps_ph = tf.placeholder(tf.int32, shape=(None), name='ps_ph')
# in_pipeline = input_pipeline(fname_ph, ps_ph)
# nodes = model(in_pipeline['img_next_op'], in_pipeline['label_next_op'])
#
#
# with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
#     sess.run([tf.global_variables_initializer(), in_pipeline['iter_init_op']], feed_dict={fname_ph: totrain_files, ps_ph: [100] * epoch_length})
#     for step in tqdm(range(epoch_length)):
#         # run train_op
#         _ = sess.run(nodes['train_op'])
#         # use saver to save weights
#         if step % epoch_length == epoch_length - 1:
#             in_dict = {
#                 'file_names': fname_ph,
#             }
#             out_dict = {
#                 'predict': nodes['predict_op'],
#                 'diff_op': nodes['diff_op']
#             }
#             tf.saved_model.simple_save(sess, './dummy/savedmodel', in_dict, out_dict)
#
# ##############################Predicting####################################
# print('*** restoring')
# # input pipeline for predict
# # create list of file names: ['test_0.h5', 'test_1.h5', ...]
# topredict_files = [os.path.join('./predict/', f) for f in os.listdir('./predict/') if f.endswith('.h5')]
# epoch_length = len(topredict_files)
#
# # save prediction images to /results folder
# if not os.path.exists('./results'):
#     os.makedirs('./results')
#
# # restore
# # set to the default graph
# graph2 = tf.Graph()
# with graph2.as_default():
#     with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
#         tf.saved_model.loader.load(
#             sess,
#             [tf.saved_model.tag_constants.SERVING], './dummy/savedmodel'
#         )
#         # import graph
#         # get operation and so on
#         file_names_ph2 = graph2.get_tensor_by_name('file_name_ph:0')
#         ps_ph2 = graph2.get_tensor_by_name('ps_ph:0')
#         predict_tensor = graph2.get_tensor_by_name('Conv1/prediction:0')
#         diff_tensor = graph2.get_tensor_by_name('Operations/difference:0')
#         iter_init_op = graph2.get_operation_by_name('predict_iter_init_op')
#
#         sess.run(iter_init_op, feed_dict={file_names_ph2: topredict_files, ps_ph2: [100] * epoch_length})
#         for step in tqdm(range(epoch_length)):
#             predict, difference = sess.run([predict_tensor, diff_tensor])
#             print(predict.shape, difference.shape)
#             with h5py.File('./results/{}.h5'.format(step), 'w') as f:
#                 a = f.create_dataset('prediction', (100, 100), dtype='float32')
#                 a[:] = predict.reshape(100, 100)
#                 b = f.create_dataset('difference', (100, 100), dtype='float32', data=difference)
#                 b[:] = difference.reshape(100, 100)
##########################
# import h5py
# import numpy as np
# import tensorflow as tf
#
# def parser(args):
#     name, patch_size = args
#     print(name)
#     name = name.decode('utf-8')
#     patch_size = int(patch_size.decode('utf-8'))
#     return name
#
#
# def _pyfn_wrapper(args):
#     return tf.py_func(parser,  #wrapped pythonic function
#                       [args],
#                       [tf.int32]  #[input, output] dtype
#                       )
#
# l_a = [i for i in range(90)]
# l_b = [10] * 90
# a = tf.placeholder(tf.int32, shape=[None])
# b = tf.placeholder(tf.int32, shape=[None])
# tmp = [(a, b)]
# print(tmp)
# ds = tf.data.Dataset.from_tensor_slices(tmp)
# ds = ds.map(_pyfn_wrapper, num_parallel_calls=5)
# ds = ds.batch(5, drop_remainder=True).shuffle(5).prefetch(5).repeat()
# it = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)  #one output with shape 1
# iter_init_op = it.make_initializer(ds, name='iter')
# a_it = it.get_next()
# sum = tf.Variable(0)
# sum = tf.add(sum, a)
#
# with tf.Session() as sess:
#     sess.run([iter_init_op, tf.global_variables_initializer()], feed_dict={a: l_a,  b: l_b})
#     # sess.run([iter_init_op])
#     for step in range(90):
#         print(sess.run(sum))

###################### save and load model with ckpt then replace input_map
# #########save part
# import tensorflow as tf
#
#
# def wrapper(x, y):
#     with tf.name_scope('wrapper'):
#         return tf.py_func(Copy, [x, y], [tf.float32, tf.float32])
#
#
# def Copy(x, y):
#     return x, y
#
#
# x_ph = tf.placeholder(tf.float32, [None], 'x_ph')
# y_ph = tf.placeholder(tf.float32, [None], 'y_ph')
#
# with tf.name_scope('input'):
#     ds = tf.data.Dataset.from_tensor_slices((x_ph, y_ph))
#     ds = ds.map(wrapper)
#     ds = ds.batch(1)
#
#     it = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
#     it_init_op = it.make_initializer(ds, name='it_init_op')
# with tf.name_scope('getnext'):
#     x_it, y_it = it.get_next()
#
# with tf.name_scope('add'):
#     V = tf.get_variable('V', [1], initializer=tf.constant_initializer(5))
#     res = tf.add(x_it, V)
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run([tf.global_variables_initializer(), it_init_op], feed_dict={y_ph: [10] * 10, x_ph: [i for i in range(10)]})
#     sess.run([res])
#     for n in tf.get_default_graph().as_graph_def().node:
#         print(n.name)
#     saver.save(sess, './dummy/ckpt/test')
# #########load part
# import tensorflow as tf
#
# def wrapper(x, y):
#     with tf.name_scope('wrapper'):
#         return tf.py_func(Copy, [x, y], [tf.float32, tf.float32])
#
#
# def Copy(x, y):
#     return x, y
#
#
# x_ph = tf.placeholder(tf.float32, [None], 'x_ph')
# y_ph = tf.placeholder(tf.float32, [None], 'y_ph')
#
# with tf.name_scope('input'):
#     ds = tf.data.Dataset.from_tensor_slices((x_ph, y_ph))
#     ds = ds.map(wrapper)
#     ds = ds.batch(1)
#
#     it = tf.data.Iterator.from_structure(ds.output_types, ds.output_shapes)
#     it_init_op = it.make_initializer(ds, name='it_init_op')
#
#
# restorer = tf.train.import_meta_graph('./dummy/ckpt/test.meta', input_map={'getnext/IteratorGetNext': tf.convert_to_tensor(it.get_next())})
# graph_def = tf.get_default_graph()
# add_op = graph_def.get_tensor_by_name('add/Add:0')
#
# for n in tf.get_default_graph().as_graph_def().node:
#     print(n.name)
#
# with tf.Session() as sess:
#     sess.run(it_init_op, feed_dict={x_ph: [i for i in range(5)], y_ph: [10] * 5})
#     restorer.restore(sess, './dummy/ckpt/test')
#
#     for _ in range(5):
#         print(sess.run([add_op]))




########################## saved_model API snippet recycled#############################################################
# prepare input dict and out dict
                    # in_dict = {
                    #     'train_files_ph': train_inputs['fnames_ph'],
                    #     'train_ps_ph': train_inputs['patch_size_ph'],
                    #     'test_files_ph': test_inputs['fnames_ph'],
                    #     'test_ps_ph': test_inputs['patch_size_ph'],
                    # }
                    # out_dict = {
                    #     'prediction': nodes['y_pred'],
                    #     'tot_op': nodes['train_op'],
                    #     'summary': nodes['summary'],
                    #     'img': nodes['img'],
                    #     'label': nodes['label']
                    # }
                    # builder
                    # tf.saved_model.simple_save(sess, './logs/{}/hour{}/savedmodel/step{}/'.format(hyperparams['date'],
                    #                                                                               hyperparams['hour'],
                    #                                                                               step + ep * hyperparams['nb_batch']), in_dict, out_dict)

########################## reproducable duplicated Adam Optimizer issue
# import tensorflow as tf
# import numpy as np
#
# X_imgs = np.asarray([np.random.rand(784).reshape(28, 28, 1) for _ in range(100)], dtype=np.float32)
# y_imgs = np.asarray([np.random.rand(784).reshape(28, 28, 1) for _ in range(100)], dtype=np.float32)
# X_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
# y_ph = tf.placeholder(tf.float32, [None, 28, 28, 1])
#
#
# with tf.name_scope("layer1"):
#     W1 = tf.get_variable("W1", shape=[3, 3, 1, 1],
#                          initializer=tf.contrib.layers.xavier_initializer())
#     b1 = tf.get_variable("b1", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
#     layer1 = tf.nn.conv2d(X_ph, W1, strides=[1, 1, 1, 1], padding='SAME') + b1
#
# with tf.name_scope("loss"):
#     loss = tf.reduce_mean(tf.losses.mean_squared_error(
#         labels=tf.cast(y_ph, tf.int32),
#         predictions=layer1))
#
# with tf.name_scope("train"):
#     optimizer = tf.train.AdamOptimizer(0.000001)
#     grads = optimizer.compute_gradients(loss)
#     train_op = optimizer.apply_gradients(grads)
#
# with tf.Session() as sess:
#     sess.run([tf.global_variables_initializer()])
#     writer = tf.summary.FileWriter('./dummy/', sess.graph, 3)
#     for i in range(100):
#         print(i)
#         sess.run(train_op, feed_dict={X_ph: X_imgs, y_ph: y_imgs})

########################20190503 try interleave_parallel
# mdl = test()
# test_train(*mdl)


# class generator_yield:
#     def __init__(self, file):
#         self.file = file
#
#     def __call__(self):
#         with h5py.File(self.file, 'r') as f:
#             yield f['X'][:], f['y'][:]
#
# def generator_return(path):
#     sess = tf.Session()
#     with sess.as_default():
#         with h5py.File(path.eval(), 'r') as f:
#             return f['X'][:], f['y'][:]
#
#
#
# dir = './proc'
# batch_size = 10000
#
# # make filenames list
# def _fnamesmaker(dir, mode='h5'):
#     fnames = []
#     for dirpath, _, filenames in os.walk(dir):
#         for fname in filenames:
#             if 'label' not in fname and fname.endswith(mode):
#                 fnames.append(os.path.abspath(os.path.join(dirpath, fname)))
#     return fnames
#
# fnames = _fnamesmaker(dir)
#
# # begin session
# with tf.Session() as sess:
#     # handle multiple files
#     # https://stackoverflow.com/questions/49579684/difference-between-dataset-from-tensors-and-dataset-from-tensor-slices
#     # fnames = tf.data.Dataset.from_tensor_slices(fnames)
#     # ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
#     #     generator(filename), tf.float32, tf.TensorShape([10000, 40, 40])), cycle_length=mp.cpu_count())
#
#     # handle multiple files (parallelized)
#     fnames = tf.data.Dataset.from_tensor_slices(fnames)
#
#     # https://www.tensorflow.org/api_docs/python/tf/contrib/data/parallel_interleave
#     # ds = fnames.apply(
#     #     tf.data.experimental.parallel_interleave(lambda filename: tf.data.Dataset.from_generator(
#     #         generator=generator_yield(filename), output_types=tf.float32,
#     #         output_shapes=tf.TensorShape([10000, 40, 40])), cycle_length=mp.cpu_count(), sloppy=False))
#     #
#     # values = ds.make_one_shot_iterator().get_next()
#     # while True:
#     #     try:
#     #         data = sess.run(values)
#     #         print(data.shape)
#     #     except tf.errors.OutOfRangeError:
#     #         print('done.')
#     #         break
#
#     # https://stackoverflow.com/questions/50046505/how-to-use-parallel-interleave-in-tensorflow
#     files = fnames.apply(tf.data.experimental.parallel_interleave(
#         generator_yield(fnames), cycle_length=mp.cpu_count(), sloppy=False))
#     files = files.cache()  # cache into memory
#     # imgs = files.map(read_decode, num_parallel_calls=mp.cpu_count())\
#     # .apply(tf.contrib.data.shuffle_and_repeat(100)) \
#     #     .batch(batch_size) \
#     #     .prefetch(5)

#######################20190221_arange main.py
# from proc import preprocess
# from train import test_train
# # from model import test
# import tensorflow as tf
# import h5py
# import os
#
# preproc = {
#     'dir': './raw',
#     'stride': 1,
#     'patch_size': 40,
#     'batch_size': 10000,
# }
#
# # preprocess(**preproc)
# # mdl = test()
# # test_train(*mdl)
#
#
# class generator_yield:
#     def __init__(self, file):
#         self.file = file
#
#     def __call__(self):
#         with h5py.File(self.file, 'r') as f:
#             yield f['X'][:], f['y'][:]
#
# def generator_return(path):
#     with h5py.File(path, 'r') as f:
#         return f['X'][:], f['y'][:]
#
#
#
# dir = './proc'
# batch_size = 10000
#
# # make filenames list
# def _fnamesmaker(dir, mode='h5'):
#     fnames = []
#     for dirpath, _, filenames in os.walk(dir):
#         for fname in filenames:
#             if 'label' not in fname and fname.endswith(mode):
#                 fnames.append(os.path.abspath(os.path.join(dirpath, fname)))
#     return fnames
#
# fnames = _fnamesmaker(dir)
# len_fnames = len(fnames)
# # begin session
# with tf.Session() as sess:
#     # handle multiple files
#     # https://stackoverflow.com/questions/49579684/difference-between-dataset-from-tensors-and-dataset-from-tensor-slices
#     # fnames = tf.data.Dataset.from_tensor_slices(fnames)
#     # ds = ds.interleave(lambda filename: tf.data.Dataset.from_generator(
#     #     generator(filename), tf.float32, tf.TensorShape([10000, 40, 40])), cycle_length=mp.cpu_count())
#
#     # handle multiple files (parallelized)
#     fnames = tf.data.Dataset.from_tensor_slices(fnames)
#     # https://stackoverflow.com/questions/50046505/how-to-use-parallel-interleave-in-tensorflow
#     # https://www.tensorflow.org/api_docs/python/tf/contrib/data/parallel_interleave
#     # ds = fnames.apply(
#     #     tf.data.experimental.parallel_interleave(lambda filename: tf.data.Dataset.from_generator(
#     #         generator=generator_yield(filename), output_types=tf.float32,
#     #         output_shapes=tf.TensorShape([10000, 40, 40])), cycle_length=mp.cpu_count(), sloppy=False))
#     #
#     # values = ds.make_one_shot_iterator().get_next()
#     # while True:
#     #     try:
#     #         data = sess.run(values)
#     #         print(data.shape)
#     #     except tf.errors.OutOfRangeError:
#     #         print('done.')
#     #         break
#
#     # https://stackoverflow.com/questions/50046505/how-to-use-parallel-interleave-in-tensorflow
#     files = fnames.apply(tf.data.experimental.parallel_interleave(lambda filename: tf.data.Dataset.from_generator(
#         generator_return, output_types=tf.float32, output_shapes=tf.TensorShape([10000, 40, 40])),
#                                                                   cycle_length=len_fnames, sloppy=False))
#     print(files)
#     files = files.cache()  # cache into memory
#     print(files)
#     # imgs = files.map(read_decode, num_parallel_calls=mp.cpu_count())\
#     # .apply(tf.contrib.data.shuffle_and_repeat(100)) \
#     #     .batch(batch_size) \
#     #     .prefetch(5)

###################20190504 arange test.py
# import tensorflow as tf
# import numpy as np
# outdir = './proc/'
# import h5py
# import os
#
#
# def nn():
#     with tf.variable_scope("placeholder"):
#         input = tf.placeholder(tf.float32, shape=[None, 10, 10])
#         y_true = tf.placeholder(tf.int32, shape=[None, 1])
#
#     with tf.variable_scope('FullyConnected'):
#         w = tf.get_variable('w', shape=[10, 10], initializer=tf.random_normal_initializer(stddev=1e-1))
#         b = tf.get_variable('b', shape=[10], initializer=tf.constant_initializer(0.1))
#         z = tf.matmul(input, w) + b
#         y = tf.nn.relu(z)
#
#         w2 = tf.get_variable('w2', shape=[10, 1], initializer=tf.random_normal_initializer(stddev=1e-1))
#         b2 = tf.get_variable('b2', shape=[1], initializer=tf.constant_initializer(0.1))
#         z = tf.matmul(y, w2) + b2
#
#     with tf.variable_scope('Loss'):
#         losses = tf.nn.sigmoid_cross_entropy_with_logits(None, tf.cast(y_true, tf.float32), z)
#         loss_op = tf.reduce_mean(losses)
#
#     with tf.variable_scope('Accuracy'):
#         y_pred = tf.cast(z > 0, tf.int32)
#         accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
#         accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")
#
#     adam = tf.train.AdamOptimizer(1e-2)
#     train_op = adam.minimize(loss_op, name="train_op")
#
#     return train_op, loss_op, accuracy
#
# def train(train_op, loss_op, accuracy):
#     with tf.Session() as sess:
#         # ... init our variables, ...
#         sess.run(tf.global_variables_initializer())
#
#         # ... check the accuracy before training (without feed_dict!), ...
#         sess.run(accuracy)
#
#         # ... train ...
#         for i in range(5000):
#             #  ... without sampling from Python and without a feed_dict !
#             _, loss = sess.run([train_op, loss_op], feed_dict={})
#
#             # We regularly check the loss
#             if i % 500 == 0:
#                 print('iter:%d - loss:%f' % (i, loss))
#
#         # Finally, we check our final accuracy
#         sess.run(accuracy)
#
# def _bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
# def parser(tfrecord):
#     img_features = tf.parse_single_example(
#         tfrecord,
#         features={
#             'X': tf.FixedLenFeature([], tf.string),
#             'y': tf.FixedLenFeature([], tf.string),
#         })
#
#     X = tf.decode_raw(img_features['X'], tf.float32)
#     y = tf.decode_raw(img_features['y'], tf.float32)
#     return X, y
#
# def tfrecordReader(filename):
#     dataset = tf.data.TFRecordDataset(filenames=filename, num_parallel_reads=10)
#     dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10, 1))
#     dataset = dataset.apply(tf.contrib.data.map_and_batch(parser, 10))
#     # dataset = dataset.prefetch(buffer_size=2)
#     return dataset
#
# class generator_yield:
#     def __init__(self, file):
#         self.file = file
#
#     def __call__(self):
#         with h5py.File(self.file, 'r') as f:
#             yield f['X'][:], f['y'][:]
#
# def input_fn(fnames, batch_size):
#     batches = fnames.apply(tf.data.experimental.parallel_interleave(lambda filename: tf.data.Dataset.from_generator(
#         generator_yield(filename), output_types=tf.float32,
#         output_shapes=tf.TensorShape([10, 10])), cycle_length=len_fnames))
#     batches.shuffle()
#     batches.batch(batch_size)
#     return batches
#
# # for i in range(5):
# #     with tf.io.TFRecordWriter(outdir + '{}_{}_{}_{}.tfrecord'.format(100, 100, 0, i)) as writer:
# #         start = 0
# #         end = 10
# #         a = np.arange(1000).reshape(10, 10, 10)
# #         for j in range(10):
# #             # Create a feature
# #             feature = {
# #                 'X': _bytes_feature(a[j, ].tostring()),
# #                 'y': _bytes_feature(a[j, ].tostring())
# #             }
# #             # Create an example protocol buffer
# #             example = tf.train.Example(features=tf.train.Features(feature=feature))
# #             # Serialize to string and write on the file
# #             writer.write(example.SerializeToString())
#
# fnames = []
# for dirpath, _, filenames in os.walk('./proc/'):
#     for fname in filenames:
#         if fname.endswith('h5'):
#             fnames.append(os.path.abspath(os.path.join(dirpath, fname)))
# len_fnames = len(fnames)
# tf.enable_eager_execution()
# fnames = tf.data.Dataset.from_tensor_slices(fnames)

################################################
#
#  build separate graph, which share variables
#
################################################
# import tensorflow as tf
# import numpy as np
# from tqdm import tqdm
# from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
# from tensorflow.python.framework import dtypes
# import os
#
# def up_2by2_ind(input_layer, ind, name=''):
#     with tf.name_scope(name):
#         in_shape = input_layer.get_shape().as_list()
#         out_shape = [tf.cast(tf.shape(input_layer), dtype=tf.int64)[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3]]
#
#         # prepare
#         _pool = tf.reshape(input_layer, [-1])
#         _range = tf.reshape(tf.range(out_shape[0], dtype=ind.dtype), [out_shape[0], 1, 1, 1])
#         tmp = tf.ones_like(ind) * _range
#         tmp = tf.reshape(tmp, [-1, 1])
#         _ind = tf.reshape(ind, [-1, 1])
#         _ind = tf.concat([tmp, _ind], 1)
#
#         # scatter
#         unpool = tf.scatter_nd(_ind, _pool, [out_shape[0], out_shape[1] * out_shape[2] * out_shape[3]])
#
#         # reshape
#         unpool = tf.reshape(unpool, out_shape)
#         return unpool
#
# def check_N_mkdir(path_to_dir):
#     if not os.path.exists(path_to_dir):
#         os.makedirs(path_to_dir, exist_ok=True)
#
# ###################### input pipeline
# def wrapper(a, b):
#     return tf.py_func(
#         wrawrapper,
#         [a, b],
#         [tf.float32, tf.float32],
#     )
#
# def wrawrapper(a, b):
#     return np.ones((1, 50, 50, 1), dtype=np.float32), np.ones((1, 50, 50, 1), dtype=np.float32)
#
# a_ph = tf.placeholder(tf.string, shape=[None], name='a_ph')
# b_ph = tf.placeholder(tf.int32, shape=[None], name='b_ph')
#
# batch = tf.data.Dataset.from_tensor_slices((a_ph, b_ph))
# batch = batch.shuffle(tf.cast(tf.shape(a_ph)[0], tf.int64))
# batch = batch.map(wrapper).prefetch(10).repeat()
# it = tf.data.Iterator.from_structure(batch.output_types, batch.output_shapes)
# iter_init_op = it.make_initializer(batch, name='iter_init_op')
# X_it, y_it = it.get_next()
#
# dropout = tf.placeholder(tf.float32, [], name='dropout')
# BN_phase = tf.placeholder(tf.bool, [], name='BN_phase')
# save_summary_step = 20
# save_model_step = 100
# check_N_mkdir('./dummy/gpus/')
# check_N_mkdir('./dummy/ckpt/')
#
# ##################### train graph on gpu1
# with tf.device('/device:GPU:0'):
#     with tf.name_scope('model'):
#         with tf.name_scope('conv'):
#             with tf.variable_scope('conv', reuse=False):
#                 w1 = tf.get_variable('w', shape=[3, 3, 1, 1], initializer=tf.initializers.glorot_normal())
#             out1 = tf.nn.conv2d(X_it, w1, strides=[1, 1, 1, 1], padding='SAME', name='conv')
#             with tf.variable_scope('conv', reuse=False):
#                 out1 = tf.layers.batch_normalization(out1, training=BN_phase, name='batch_norm')
#             out1 = tf.nn.relu(out1, 'relu')
#             out1, ind1 = tf.nn.max_pool_with_argmax(out1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
#         with tf.name_scope('dnn'):
#             flat = tf.reshape(out1, [1, 625])
#             with tf.variable_scope('dnn2', reuse=False):
#                 w2 = tf.get_variable('w2', shape=[625, 625], initializer=tf.initializers.glorot_normal())
#             dnn_out = tf.matmul(flat, w2)
#             dnn_out = tf.nn.dropout(dnn_out, keep_prob=dropout, name='do')
#             dnn_out = tf.nn.relu(dnn_out, name='relu')
#             dnn_out = tf.reshape(dnn_out, shape=[1, 25, 25, 1], name='dnn')
#
#         with tf.name_scope('deconv'):
#             out1 = up_2by2_ind(dnn_out, ind1, 'up1')
#             with tf.variable_scope('deconv', reuse=False):
#                 w3 = tf.get_variable('w', shape=[3, 3, 1, 1], initializer=tf.initializers.glorot_normal())
#             out1 = tf.nn.conv2d(out1, w3, strides=[1, 1, 1, 1], padding='SAME', name='deconv')
#             with tf.variable_scope('deconv', reuse=False):
#                 out1 = tf.layers.batch_normalization(out1, training=BN_phase, name='batch_norm')
#             logits = tf.nn.relu(out1, 'logits')
#
#         #todo: here with tabulation with tf.name_scope('operation'):
# with tf.name_scope('loss'):
#     mse = tf.losses.mean_squared_error(labels=y_it, predictions=logits)
#
# with tf.device('/device:GPU:0'):
#     with tf.name_scope('operation'):
#         opt = tf.train.AdamOptimizer(learning_rate=0.0001, name='Adam')
#         grads = opt.compute_gradients(mse)
#         train_op = opt.apply_gradients(grads, name='apply_grad')
#
# # note: like following the gradient will not be in GPU:0
# #  opt = tf.train.AdamOptimizer(learning_rate=0.0001, name='Adam')
# #  grads = opt.compute_gradients(mse)
# #  train_op = opt.apply_gradients(grads, name='apply_grad')
#
# with tf.name_scope('train_metrics'):
#     acc_val_op, acc_update_op = tf.metrics.accuracy(labels=y_it, predictions=logits)
#     summ_acc = tf.summary.merge([tf.summary.scalar('accuracy', acc_val_op)])
#     grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
#
# with tf.name_scope('train_summary'):
#     merged = tf.summary.merge([summ_acc, grad_sum, tf.summary.histogram("weights", w1)])
#
# ###################### test graph on gpu2
# with tf.device('/device:GPU:1'):
#     with tf.name_scope('model'):
#         with tf.name_scope('conv'):
#             with tf.variable_scope('conv', reuse=True):
#                 w1 = tf.get_variable('w', shape=[3, 3, 1, 1], initializer=tf.initializers.glorot_normal())
#             out1 = tf.nn.conv2d(X_it, w1, strides=[1, 1, 1, 1], padding='SAME', name='conv')
#             with tf.variable_scope('conv', reuse=True):
#                 out1 = tf.layers.batch_normalization(out1, training=BN_phase, name='batch_norm')
#             out1 = tf.nn.relu(out1, 'relu')
#             out1, ind1 = tf.nn.max_pool_with_argmax(out1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
#         with tf.name_scope('dnn'):
#             flat = tf.reshape(out1, [1, 625])
#             with tf.variable_scope('dnn2', reuse=True):
#                 w2 = tf.get_variable('w2', shape=[625, 625], initializer=tf.initializers.glorot_normal())
#             dnn_out = tf.matmul(flat, w2)
#             dnn_out = tf.nn.dropout(dnn_out, keep_prob=dropout, name='do')
#             dnn_out = tf.nn.relu(dnn_out, name='relu')
#             dnn_out = tf.reshape(dnn_out, shape=[1, 25, 25, 1], name='dnn')
#
#         with tf.name_scope('deconv'):
#             out1 = up_2by2_ind(dnn_out, ind1, 'up1')
#             with tf.variable_scope('deconv', reuse=True):
#                 w3 = tf.get_variable('w', shape=[3, 3, 1, 1], initializer=tf.initializers.glorot_normal())
#             out1 = tf.nn.conv2d(out1, w3, strides=[1, 1, 1, 1], padding='SAME', name='deconv')
#             with tf.variable_scope('conv', reuse=True):
#                 out1 = tf.layers.batch_normalization(out1, training=BN_phase, name='batch_norm')
#             logits = tf.nn.relu(out1, 'logits')
#
#
# with tf.name_scope('test_metrics'):
#     acc_val_op2, acc_update_op2 = tf.metrics.accuracy(labels=y_it, predictions=logits)
#     summ_acc2 = tf.summary.merge([tf.summary.scalar('accuracy', acc_val_op2)])
#
# with tf.name_scope('test_summary'):
#     merged2 = tf.summary.merge([summ_acc2, tf.summary.histogram("weights", w1), tf.summary.histogram("weights", w2)])
#
#
# ##############################################
# with tf.Session() as sess:
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), iter_init_op],
#              feed_dict={a_ph: ['a'], b_ph: [10]})
#     model_saver = tf.train.Saver(max_to_keep=100000)
#     train_writer = tf.summary.FileWriter('./dummy/gpus/train/', sess.graph)
#     test_writer = tf.summary.FileWriter('./dummy/gpus/test/', sess.graph)
#     for i in tqdm(range(1000)):
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         if i % save_summary_step == 0:
#             _, rlt, summary, _, _ = sess.run([train_op, out1, merged, acc_update_op, update_ops], feed_dict={
#                 dropout: 0.1,
#                 BN_phase: True,
#             }
#                                           )
#             # print('train:', rlt)
#             train_writer.add_summary(summary, global_step=i)
#         else:
#             _, rlt, _ = sess.run([train_op, out1, update_ops], feed_dict={dropout: 1, BN_phase: True})
#             # print('train:', rlt)
#         if i % save_model_step == 0:
#             model_saver.save(sess, './dummy/ckpt/step{}'.format(i))
#             if i != 0:
#                 for j in tqdm(range(5)):
#                     rlt, summary = sess.run([logits, merged2], feed_dict={
#                         dropout: 1,
#                         BN_phase: False,
#                     }
#                                             )
#                     # print('test:', rlt)
#                     test_writer.add_summary(summary, global_step=j)
#
# ######################## inference and optimize pb
# def freeze_ckpt_for_inference(ckpt_path=None, conserve_nodes=None):
#     # clean graph first
#     tf.reset_default_graph()
#     # freeze ckpt then convert to pb
#     new_input = tf.placeholder(tf.float32, shape=[None, 50, 50, 1], name='new_input')
#     new_BN = tf.placeholder(tf.bool, name='new_BN')
#     new_dropout = tf.placeholder(tf.float32, name='new_dropout')
#
#     restorer = tf.train.import_meta_graph(
#         ckpt_path + '.meta',
#         input_map={
#             'IteratorGetNext': new_input,
#             'BN_phase': new_BN,
#             'dropout': new_dropout,
#         },
#         clear_devices=True,
#     )
#
#     input_graph_def = tf.get_default_graph().as_graph_def()
#     check_N_mkdir('./dummy/pb/')
#     check_N_mkdir('./dummy/tb/')
#
#     # freeze to pb
#     with tf.Session() as sess:
#         # restore variables
#         restorer.restore(sess, './dummy/ckpt/step900')
#         # convert variable to constant
#         output_graph_def = tf.graph_util.convert_variables_to_constants(
#             sess=sess,
#             input_graph_def=input_graph_def,
#             output_node_names=conserve_nodes,
#         )
#
#         # save to pb
#         with tf.gfile.GFile('./dummy/pb/freeze.pb', 'wb') as f:  # 'wb' stands for write binary
#             f.write(output_graph_def.SerializeToString())
#
#
# def optimize_graph_for_inference(pb_dir=None, conserve_nodes=None):
#     tf.reset_default_graph()
#     check_N_mkdir(pb_dir)
#
#     # import pb file
#     with tf.gfile.FastGFile(pb_dir + 'freeze.pb', "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#     # optimize graph
#     optimize_graph_def = optimize_for_inference(input_graph_def=graph_def,
#                                                 input_node_names=['new_input', 'new_BN', 'new_dropout'],
#                                                 output_node_names=conserve_nodes,
#                                                 placeholder_type_enum=[dtypes.float32.as_datatype_enum,
#                                                                        dtypes.bool.as_datatype_enum,
#                                                                        dtypes.float32.as_datatype_enum,
#                                                                        ]
#                            )
#     with tf.gfile.GFile(pb_dir + 'optimize.pb', 'wb') as f:
#         f.write(optimize_graph_def.SerializeToString())
#
#
# def visualize_pb_file(pb_path):
#     tf.reset_default_graph()
#
#     with tf.gfile.FastGFile(pb_path, "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#     #fixme: ValueError: NodeDef expected inputs '' do not match 1 inputs specified; Op<name=Const; signature= -> output:dtype; attr=value:tensor; attr=dtype:type>; NodeDef: {{node import/model/conv1/batch_norm/cond/Const}}
#     # https://github.com/tensorflow/tensorflow/issues/19838
#     # solution: https://github.com/tensorflow/tensorflow/issues/19838#issuecomment-559775353
#     tf.graph_util.import_graph_def(
#         graph_def,
#     )
#
#     with tf.Session() as sess:
#         tf.summary.FileWriter('./dummy/tb/optimize', sess.graph)
#
# conserve_nodes = ['model/deconv/logits']
# freeze_ckpt_for_inference(ckpt_path='./dummy/ckpt/step900', conserve_nodes=conserve_nodes)
# optimize_graph_for_inference(pb_dir='./dummy/pb/', conserve_nodes=conserve_nodes)
# visualize_pb_file('./dummy/pb/optimize.pb')


###################################################
#
#       dice loss function for classfication
#
###################################################
# import tensorflow as tf
# from tqdm import tqdm
# import numpy as np
# import os
#
#
# def up_2by2_ind(input_layer, ind, name=''):
#     with tf.name_scope(name):
#         in_shape = input_layer.get_shape().as_list()
#         out_shape = [tf.cast(tf.shape(input_layer), dtype=tf.int64)[0], in_shape[1] * 2, in_shape[2] * 2, in_shape[3]]
#
#         # prepare
#         _pool = tf.reshape(input_layer, [-1])
#         _range = tf.reshape(tf.range(out_shape[0], dtype=ind.dtype), [out_shape[0], 1, 1, 1])
#         tmp = tf.ones_like(ind) * _range
#         tmp = tf.reshape(tmp, [-1, 1])
#         _ind = tf.reshape(ind, [-1, 1])
#         _ind = tf.concat([tmp, _ind], 1)
#
#         # scatter
#         unpool = tf.scatter_nd(_ind, _pool, [out_shape[0], out_shape[1] * out_shape[2] * out_shape[3]])
#
#         # reshape
#         unpool = tf.reshape(unpool, out_shape)
#         return unpool
#
#
# def check_N_mkdir(path_to_dir):
#     if not os.path.exists(path_to_dir):
#         os.makedirs(path_to_dir, exist_ok=True)
#
#
# def wrapper(a, b):
#     return tf.py_func(
#         wrawrapper,
#         [a, b],
#         [tf.float32, tf.int8],
#     )
#
# rand = np.random.randint(0, 1, size=(8, 50, 50, 3), dtype=np.int8)
# def wrawrapper(a, b):
#     return np.ones((8, 50, 50, 1), dtype=np.float32), rand
#
#
# def dice(y_true, y_pred):
#     # [batch_size, height, weight, channel]
#     # note: take only height and weight
#     axis = (1, 2, 3)
#     #  minimize even equally for all classes (even for minor class)
#     y_true = tf.cast(y_true, tf.float32)
#     numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=axis)
#     denominator = tf.reduce_sum(y_true + y_pred, axis=axis)
#
#     return 1 - (numerator) / (denominator)
#
#
# a_ph = tf.placeholder(tf.string, shape=[None], name='a_ph')
# b_ph = tf.placeholder(tf.int32, shape=[None], name='b_ph')
#
# batch = tf.data.Dataset.from_tensor_slices((a_ph, b_ph))
# batch = batch.shuffle(tf.cast(tf.shape(a_ph)[0], tf.int64))
# batch = batch.map(wrapper).prefetch(10).repeat()
# it = tf.data.Iterator.from_structure(batch.output_types, batch.output_shapes)
# iter_init_op = it.make_initializer(batch, name='iter_init_op')
# X_it, y_it = it.get_next()
#
# dropout = tf.placeholder(tf.float32, [], name='dropout')
# BN_phase = tf.placeholder(tf.bool, [], name='BN_phase')
# save_summary_step = 20
# save_model_step = 100
# check_N_mkdir('./dummy/gpus/')
# check_N_mkdir('./dummy/ckpt/')
#
# ##################### train graph on gpu1
#
# with tf.name_scope('model'):
#     with tf.name_scope('conv'):
#         with tf.variable_scope('conv', reuse=False):
#             w1 = tf.get_variable('w', shape=[3, 3, 1, 3], initializer=tf.initializers.glorot_normal())
#         out1 = tf.nn.conv2d(X_it, w1, strides=[1, 1, 1, 1], padding='SAME', name='conv')
#         with tf.variable_scope('conv', reuse=False):
#             out1 = tf.layers.batch_normalization(out1, training=BN_phase, name='batch_norm')
#         out1 = tf.nn.relu(out1, 'relu')
#         out1, ind1 = tf.nn.max_pool_with_argmax(out1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='maxpool')
#     with tf.name_scope('dnn'):
#         flat = tf.reshape(out1, [-1, 625 * 3])
#         with tf.variable_scope('dnn2', reuse=False):
#             w2 = tf.get_variable('w2', shape=[625 * 3, 625 * 3], initializer=tf.initializers.glorot_normal())
#         dnn_out = tf.matmul(flat, w2)
#         dnn_out = tf.nn.dropout(dnn_out, keep_prob=dropout, name='do')
#         dnn_out = tf.nn.relu(dnn_out, name='relu')
#         dnn_out = tf.reshape(dnn_out, shape=[-1, 25, 25, 3], name='dnn')
#
#     with tf.name_scope('deconv'):
#         out1 = up_2by2_ind(dnn_out, ind1, 'up1')
#         with tf.variable_scope('deconv', reuse=False):
#             w3 = tf.get_variable('w', shape=[3, 3, 3, 3], initializer=tf.initializers.glorot_normal())
#         out1 = tf.nn.conv2d(out1, w3, strides=[1, 1, 1, 1], padding='SAME', name='deconv')
#         with tf.variable_scope('deconv', reuse=False):
#             out1 = tf.layers.batch_normalization(out1, training=BN_phase, name='batch_norm')
#         logits = tf.nn.relu(out1, 'logits')
#
#
# with tf.name_scope('loss'):
#     DSC = dice(y_true=y_it, y_pred=logits)
#
# with tf.name_scope('operation'):
#     opt = tf.train.AdamOptimizer(learning_rate=0.0001, name='Adam')
#     grads = opt.compute_gradients(DSC)
#     train_op = opt.apply_gradients(grads, name='apply_grad')
#
# with tf.name_scope('train_metrics'):
#     acc_val_op, acc_update_op = tf.metrics.accuracy(labels=y_it, predictions=logits)
#     lss_val_op, lss_update_op = tf.metrics.mean(DSC)
#     summ_acc = tf.summary.merge([tf.summary.scalar('accuracy', acc_val_op)])
#     summ_lss = tf.summary.merge([tf.summary.scalar('loss', lss_val_op)])
#     grad_sum = tf.summary.merge([tf.summary.histogram('{}/grad'.format(g[1].name), g[0]) for g in grads])
#
# with tf.name_scope('train_summary'):
#     merged = tf.summary.merge([summ_acc, summ_lss, grad_sum, tf.summary.histogram("weights", w1)])
#
#
# with tf.Session() as sess:
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer(), iter_init_op],
#              feed_dict={a_ph: ['a'], b_ph: [10]})
#     model_saver = tf.train.Saver(max_to_keep=100000)
#     train_writer = tf.summary.FileWriter('./dummy/gpus/train/', sess.graph)
#     test_writer = tf.summary.FileWriter('./dummy/gpus/test/', sess.graph)
#     for i in tqdm(range(100)):
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         if i % save_summary_step == 0:
#             _, rlt, summary, _, _, _ = sess.run([train_op, logits, merged, acc_update_op, lss_update_op, update_ops], feed_dict={
#                 dropout: 0.1,
#                 BN_phase: True,
#             })
#
#             print('train:', rlt)
#             train_writer.add_summary(summary, global_step=i)
#         else:
#             _, rlt, _ = sess.run([train_op, logits, update_ops], feed_dict={dropout: 1, BN_phase: True})
#             print('train:', rlt)

############################################
#
#    verify batch norm in inference
#
############################################

# import tensorflow as tf
# from tqdm import tqdm
# import numpy as np
# from util import print_nodes_name_shape, check_N_mkdir
# from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
# from tensorflow.python.framework import dtypes
#
# inputs = np.ones((8, 2, 2, 1))
# outputs = np.arange(8 * 2 * 2 * 3).reshape((8, 2, 2, 3))
# input_ph = tf.placeholder(tf.float32, shape=(None, 2, 2, 1), name='input_ph')
# output_ph = tf.placeholder(tf.float32, shape=(None, 2, 2, 3), name='output_ph')
# is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
#
# # build a one layer Full layer with BN and save 2 ckpts
# with tf.name_scope('model'):
#     out = tf.reshape(input_ph, shape=(-1, 2 * 2 * 1), name='flatten')
#     with tf.variable_scope('dnn1', reuse=False):
#         w1 = tf.get_variable('w1', dtype=tf.float32, shape=[4 * 1, 4 * 3], initializer=tf.initializers.glorot_normal())
#         b1 = tf.get_variable('b1', dtype=tf.float32, shape=[4 * 3], initializer=tf.initializers.glorot_normal())
#     out = tf.matmul(out, w1) + b1
#     out = tf.layers.batch_normalization(out, training=is_training, name='BN')
#     logits = tf.nn.relu(out)
#     logits = tf.reshape(logits, shape=(-1, 2, 2, 3))
#
# with tf.name_scope('loss'):
#     MSE = tf.losses.mean_squared_error(labels=output_ph, predictions=logits)
#
# with tf.name_scope('operations'):
#     opt = tf.train.AdamOptimizer(learning_rate=0.0001, name='Adam')
#     grads = opt.compute_gradients(MSE)
#     train_op = opt.apply_gradients(grads, name='apply_grad')
#
# # train
# with tf.Session() as sess:
#     # prepare
#     graph = tf.get_default_graph()
#     print_nodes_name_shape(graph)
#     saver = tf.train.Saver()
#
#     # init variables
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     saver.save(sess, './dummy/ckpt/step0')
#     # train
#     for i in tqdm(range(100)):
#         with tf.variable_scope('', reuse=True):
#             mov_avg, mov_std, beta, gamma = sess.run([tf.get_variable('BN/moving_mean'),
#                                          tf.get_variable('BN/moving_variance'),
#                                                       tf.get_variable('BN/beta'),
#                                                       tf.get_variable('BN/gamma')])
#             print('\nmov_avg: {}, \nmov_std: {}, \nbeta: {}, \ngamma: {}'.format(mov_avg, mov_std, beta, gamma))
#         _, _ = sess.run([train_op, tf.get_collection(tf.GraphKeys.UPDATE_OPS)], feed_dict={
#             input_ph: inputs,
#             output_ph: outputs,
#             is_training: True,
#         })
#
#     for i in tqdm(range(100)):
#         with tf.variable_scope('', reuse=True):
#             mov_avg, mov_std, beta, gamma = sess.run([tf.get_variable('BN/moving_mean'),
#                                          tf.get_variable('BN/moving_variance'),
#                                                       tf.get_variable('BN/beta'),
#                                                       tf.get_variable('BN/gamma')])
#             print('\nmov_avg: {}, \nmov_std: {}, \nbeta: {}, \ngamma: {}'.format(mov_avg, mov_std, beta, gamma))
#         _ = sess.run([graph.get_tensor_by_name('model/Reshape:0')], feed_dict={
#             input_ph: inputs,
#             is_training: False,
#         })
#
#         # print moving avg/std
#     saver.save(sess, './dummy/ckpt/step100')
#
#
# def freeze_ckpt_for_inference(ckpt_path=None, conserve_nodes=None):
#     # clean graph first
#     tf.reset_default_graph()
#     # freeze ckpt then convert to pb
#     new_input = tf.placeholder(tf.float32, shape=[None, 10, 10, 1], name='new_input')
#     new_is_training = tf.placeholder(tf.bool, name='new_is_training')
#
#     restorer = tf.train.import_meta_graph(
#         ckpt_path + '.meta',
#         input_map={
#             'input_ph': new_input,
#             'is_training': new_is_training,
#         },
#         clear_devices=True,
#     )
#
#     input_graph_def = tf.get_default_graph().as_graph_def()
#     check_N_mkdir('./dummy/pb/')
#     check_N_mkdir('./dummy/tb/')
#
#     # freeze to pb
#     with tf.Session() as sess:
#         # restore variables
#         restorer.restore(sess, './dummy/ckpt/step100')
#         # convert variable to constant
#         output_graph_def = tf.graph_util.convert_variables_to_constants(
#             sess=sess,
#             input_graph_def=input_graph_def,
#             output_node_names=conserve_nodes,
#         )
#
#         # save to pb
#         with tf.gfile.GFile('./dummy/pb/freeze.pb', 'wb') as f:  # 'wb' stands for write binary
#             f.write(output_graph_def.SerializeToString())
#
#
# def optimize_graph_for_inference(pb_dir=None, conserve_nodes=None):
#     tf.reset_default_graph()
#     check_N_mkdir(pb_dir)
#
#     # import pb file
#     with tf.gfile.FastGFile(pb_dir + 'freeze.pb', "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#     # optimize graph
#     optimize_graph_def = optimize_for_inference(input_graph_def=graph_def,
#                                                 input_node_names=['new_input', 'new_is_training'],
#                                                 output_node_names=conserve_nodes,
#                                                 placeholder_type_enum=[dtypes.float32.as_datatype_enum,
#                                                                        dtypes.bool.as_datatype_enum,
#                                                                        dtypes.float32.as_datatype_enum,
#                                                                        ]
#                            )
#     with tf.gfile.GFile(pb_dir + 'optimize.pb', 'wb') as f:
#         f.write(optimize_graph_def.SerializeToString())
#
# conserve_nodes = ['model/Reshape']
# freeze_ckpt_for_inference(ckpt_path='./dummy/ckpt/step100', conserve_nodes=conserve_nodes)
# optimize_graph_for_inference(pb_dir='./dummy/pb/', conserve_nodes=conserve_nodes)
#
# # cleaning
# tf.reset_default_graph()
#
# # load pb
# with tf.gfile.FastGFile('./dummy/pb/optimize.pb', "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#
# print('\n Now inference*******************************')
# # inference
# with tf.Session() as sess:
#     tf.graph_util.import_graph_def(
#         graph_def,
#     )
#     # prepare
#     G = tf.get_default_graph()
#     print_nodes_name_shape(G)
#     new_input = G.get_tensor_by_name('import/new_input:0')
#     new_is_training = G.get_tensor_by_name('import/new_is_training:0')
#     new_output = G.get_tensor_by_name('import/' + conserve_nodes[-1] + ':0')
#
#     # train
#     for i in tqdm(range(100)):
#         # print moving avg/std
#         with tf.variable_scope('', reuse=True):
#             mov_avg, mov_std, beta, gamma = sess.run([G.get_tensor_by_name('import/BN/moving_mean:0'),
#                                                       G.get_tensor_by_name('import/BN/moving_variance:0'),
#                                                       G.get_tensor_by_name('import/BN/beta:0'),
#                                                       G.get_tensor_by_name('import/BN/gamma:0')])
#             print('\nmov_avg: {}, \nmov_std: {}, \nbeta: {}, \ngamma: {}'.format(mov_avg, mov_std, beta, gamma))
#         new_out = sess.run([new_output], feed_dict={
#             new_input: inputs,
#             new_output: outputs,
#             new_is_training: False,
#         })
#         # print('out: {}'.format(new_out))

########################################
#
#      Softmax vs sparse softmax
#
########################################
#
# # https://stackoverflow.com/a/43577900/9217178
# import tensorflow as tf
# from random import randint
#
# dims = 8
# pos = randint(0, dims - 1)
#
# logits = tf.random_uniform([dims], maxval=3, dtype=tf.float32)
# labels = tf.one_hot(pos, dims)
#
# res1 = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
# res2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.constant(pos))
#
# with tf.Session() as sess:
#     a, b = sess.run([res1, res2])
#     print(a, b)
#     print(a == b)

####################################
#
#      Loss landscape
#
####################################
# import tensorflow as tf
# import numpy as np
# from tqdm import tqdm
# from util import check_N_mkdir, print_nodes_name_shape
# import h5py as h5
# import copy
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.interpolate import interp2d
# import os
# prevent GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#
# #note: state includes weights, bias, BN
# def get_diff_state(state1, state2):
#     assert isinstance(state1, dict)
#     assert isinstance(state2, dict)
#     return {k: v2 - v1 for (k, _), (v1, v2) in zip(state1.items(), state2.items())}
#
#
# def get_random_state(state):
#     assert isinstance(state, dict)
#     return {k: np.random.randn(*v.shape) for k, v in state.items()}
#
#
# def get_state(ckpt_path):
#     tf.reset_default_graph()
#     loader = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
#     state = {}
#     with tf.Session() as sess:
#         loader.restore(sess, ckpt_path)
#         # print_nodes_name_shape(tf.get_default_graph())
#         ckpt_weight_names = []
#         for node in tf.get_default_graph().as_graph_def().node:
#             if node.name.endswith('w1') or \
#                     node.name.endswith('b1') or \
#                     node.name.endswith('beta') or \
#                     node.name.endswith('gamma'):
#                 ckpt_weight_names.append(node.name + ':0')
#
#         # get weights/bias
#         for k in ckpt_weight_names:
#             v = sess.run(k)
#             state[k] = v
#     return state
#
#
# def _normalize_direction(perturbation, weight):
#     assert isinstance(perturbation, np.ndarray)
#     assert isinstance(weight, np.ndarray)
#     norm_w = np.linalg.norm(weight)
#     norm_pert = np.linalg.norm(perturbation)
#     print('norm_w, norm_pert: {}, {}'.format(norm_w, norm_pert))
#     perturbation *= norm_w / (norm_pert)
#     return perturbation, weight
#
#
# def normalize_state(directions, state):
#     assert isinstance(directions, dict)
#     assert isinstance(state, dict)
#     backup = copy.deepcopy(directions)
#     for (d_name, direct), (name, weight) in zip(backup.items(), state.items()):
#         _normalize_direction(direct, weight)
#     return backup
#
#
# def move_state(sess, name, value, leap):
#     # if ('beta' not in name) and ('gamma' not in name):
#     weight = sess.graph.get_tensor_by_name(name)
#     try:
#         # print('leap: {}'.format(leap))
#         # initial = sess.run(weight)
#         # print('\n{} init: {}'.format(name, initial))
#         # assign = sess.run(tf.assign(weight, value))
#         # print('\n{} Assigned: {}'.format(name, assign))
#         after = sess.run(tf.assign(weight, value + leap))
#         # print('\nAfter: {}'.format(after))
#     except Exception as e:
#         print('initial shape: \n', sess.run(value).shape)
#         print('leap shape: \n', leap.shape)
#         print('\nError threw while trying to move weight: {}'.format(name))
#         print(e)
#
#
#
# def feed_forward(sess, graph, state, direction_2D, xcoord, ycoord, inputs, outputs, comm=None):
#     '''return loss and acc'''
#     assert isinstance(direction_2D, list), 'dir should be list'
#     assert isinstance(xcoord, float), 'xcoord should be float'
#     assert isinstance(ycoord, float), 'ycoord should be float'
#     # print('inputs: {}'.format(inputs))
#     # print('outputs: {}'.format(outputs))
#     # print('xcoord:', xcoord)
#     sess.run([tf.local_variables_initializer()])  # note: should initialize here otherwise it will keep cumulating for the average
#     # change state in the neural network
#     new_logits = graph.get_tensor_by_name('model/MLP/logits:0')
#     loss_tensor = graph.get_tensor_by_name('metrics/loss/value:0')
#     acc_tensor = graph.get_tensor_by_name('metrics/acc/value:0')
#     new_loss_update_op = graph.get_operation_by_name('metrics/loss/update_op')
#     new_acc_update_op = graph.get_operation_by_name('metrics/acc/update_op')
#     new_input_ph = graph.get_tensor_by_name('input_ph:0')
#     new_output_ph = graph.get_tensor_by_name('output_ph:0')
#     new_BN_ph = graph.get_tensor_by_name('BN_phase:0')
#
#     # print('inputs avg: {}, outputs avg: {}'.format(np.mean(inputs), np.mean(outputs)))
#
#     dx = {k: xcoord * v for k, v in direction_2D[0].items()}  # step size * direction x
#     dy = {k: ycoord * v for k, v in direction_2D[1].items()}  # step size * direction y
#     change = {k: _dx + _dy for (k, _dx), (_, _dy) in zip(dx.items(), dy.items())}
#     # calculate the perturbation
#     for k, v in state.items():
#         move_state(sess, name=k, value=v, leap=change[k])
#
#     # feed forward with batches
#     for repeat in range(2):
#         #note: (TF intrisic: at least 2 times, or loss/acc 0.0)should iterate at least several time, Or loss=acc=0, since there's a counter for the average
#         new_log, loss, acc, _, _ = sess.run([new_logits, loss_tensor, acc_tensor, new_acc_update_op, new_loss_update_op],
#                                    feed_dict={new_input_ph: inputs,
#                                               new_output_ph: outputs,
#                                               new_BN_ph: True,
#                                               #note: (TF1.14)WTF? here should be True while producing loss-landscape
#                                               #fixme: should check if the mov_avg/mov_std/beta/gamma change
#                                               })
#     if comm is not None:
#         if comm.Get_rank() != 0:
#             comm.send(1, dest=0, tag=tag_compute)
#
#     # print('lss:{}, acc:{}, predict:{}'.format(loss, acc, np.mean(new_log)))
#     return loss, acc
#
#
# def csv_interp(x_mesh, y_mesh, metrics_tensor, out_path, interp_scope=5):
#     new_xmesh = np.linspace(np.min(x_mesh), np.max(x_mesh), interp_scope * x_mesh.shape[0])
#     new_ymesh = np.linspace(np.min(y_mesh), np.max(y_mesh), interp_scope * x_mesh.shape[1])
#     newxx, newyy = np.meshgrid(new_xmesh, new_ymesh)
#
#     # interpolation
#     interpolation = interp2d(x_mesh, y_mesh, metrics_tensor, kind='cubic')
#     zval = interpolation(new_xmesh, new_ymesh)
#     pd.DataFrame({'xcoord': newxx.ravel(),
#                   'ycoord': newyy.ravel(),
#                   'zval': zval.ravel()}
#                  ).to_csv(out_path, index=False)
#
#
# config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 1})

################## model
# inputs = np.random.randn(8, 20, 20, 1) + np.ones((8, 20, 20, 1)) * 3  # noise + avg:3
# outputs = inputs ** 2  # avg: 9

##########################
#
#   note: uncomment below to train a model
#
##########################
# input_ph = tf.placeholder(tf.float32, [None, 20, 20, 1], name='input_ph')
# output_ph = tf.placeholder(tf.float32, [None, 20, 20, 1], name='output_ph')
# BN_phase = tf.placeholder(tf.bool, [], name='BN_phase')
# # model
#
# with tf.variable_scope('model'):
#     with tf.name_scope('MLP'):
#         w1 = tf.get_variable('w1', [400 * 1, 400 * 1])
#         # b1 = tf.get_variable('b1', [400 * 1])
#         flatten = tf.reshape(input_ph, [-1, 400])
#         layer = tf.matmul(flatten, w1)
#         # layer = layer + b1
#         layer = tf.layers.batch_normalization(layer, training=BN_phase)
#         logits = tf.nn.relu(layer)
#         logits = tf.reshape(logits, [-1, 20, 20, 1], name='logits')
#
# with tf.name_scope('operation'):
#     loss = tf.losses.mean_squared_error(labels=output_ph, predictions=logits, scope='MSE')
#     opt = tf.train.AdamOptimizer(learning_rate=1)
#     grads = opt.compute_gradients(loss)
#     train_op = opt.apply_gradients(grads)
#
# with tf.name_scope('metrics'):
#     loss_val_op, loss_update_op = tf.metrics.mean(loss, name='loss')
#     acc_val_op, acc_update_op = tf.metrics.accuracy(labels=tf.cast(output_ph, tf.int32), predictions=tf.cast(logits, tf.int32), name='acc')
#
#
# ################## trainning
# check_N_mkdir('./dummy/ckpt/')
#
# with tf.Session() as sess:
#     saver = tf.train.Saver(max_to_keep=10000)
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     saver.save(sess, './dummy/ckpt/step0')
#     dict = {'step':[], 'loss': [], 'accuracy': []}
#     for step in tqdm(range(10000)):
#         inputs = np.random.randn(8, 20, 20, 1) + np.ones((8, 20, 20, 1)) * 8  # noise + avg:3
#         outputs = np.random.randn(8, 20, 20, 1) + np.ones((8, 20, 20, 1)) * 64  # avg: 9
#         _, logit, _acc, _loss, _, _, = sess.run(
#             [train_op, logits, acc_val_op, loss_val_op, acc_update_op, loss_update_op], feed_dict={
#                 input_ph: inputs,
#                 output_ph: outputs,
#                 BN_phase: True,
#             }
#         )
#         print('acc {} lss {} log {}'.format(_acc, _loss, np.mean(logit)))
#         dict['loss'].append(_loss)
#         dict['accuracy'].append(_acc)
#         dict['step'].append(step)
#         # note: acc 0.004492355510592461 lss 4655.95654296875 log -22.420454025268555 (with b1)
#         #  acc 0.1409694403409958 lss 12.473745346069336 log 9.876103401184082 (with BN)
#         #  acc 0.1338576078414917 lss 13.37901782989502 log 9.91382122039795(with BN then b1)
#         #  acc 0.14119061827659607 lss 12.39465045928955 log 10.003917694091797 (with b1 then BN)
#         #  acc acc 0.1074141189455986 lss 21.29730224609375 log 9.910054206848145(with BN, lr=5)
#         if step % 1000 == 0:
#             saver.save(sess, './dummy/ckpt/step{}'.format(step))
#     pd.DataFrame(dict).to_csv('./dummy/loss_acc.csv', index=False, sep=';')

##########################
#
#   note: uncomment above to train a model
#
##########################


################## restore 2 ckpts and calculate directions

# # get 2 ckpts
# ckpt1_path = './dummy/ckpt/step0'
# ckpt2_path = './dummy/ckpt/step4999'
# ckpt3_path = './dummy/ckpt/step9999'
#
# # get state 1, 2
# state1 = get_state(ckpt1_path)
# state2 = get_state(ckpt2_path)
# state3 = get_state(ckpt3_path)
#
# # get random directions
# rand_direct1 = get_random_state(state1)
# rand_direct1_bis = get_random_state(state1)
# rand_direct2 = get_random_state(state2)
# rand_direct2_bis = get_random_state(state2)
# rand_direct3 = get_random_state(state3)
# rand_direct3_bis = get_random_state(state3)
#
# # normalize directions by weights
# normalized_ds1 = normalize_state(rand_direct1, state1)
# normalized_ds1_bis = normalize_state(rand_direct1_bis, state1)
# normalized_ds2 = normalize_state(rand_direct2, state2)
# normalized_ds2_bis = normalize_state(rand_direct2_bis, state2)
# normalized_ds3 = normalize_state(rand_direct3, state3)
# normalized_ds3_bis = normalize_state(rand_direct3_bis, state3)

# create direction and surface .h5
# loss_landscape_file_path = './dummy/loss_land.h5'
# x_min, x_max, x_nb = -1, 1, 51
# y_min, y_max, y_nb = -1, 1, 51
# xcoord = np.linspace(x_min, x_max, x_nb).ravel()
# ycoord = np.linspace(y_min, y_max, y_nb).ravel()
# xm, ym = np.meshgrid(xcoord, ycoord)




# calculate loss/acc for each point on the surface (try first with only for loop)
# start feeding


# l_states = [
#     state3,
#     state2,
#     state1
# ]
# l_ckpts = [
#     ckpt3_path,
#     ckpt2_path,
#     ckpt1_path
# ]
# l_steps = [
#     '9999',
#     '4999',
#     '0'
# ]
#
# l_directions = [
#     normalized_ds3,
#     normalized_ds2,
#     normalized_ds1,
# ]
#
# l_directions_bis = [
#     normalized_ds3_bis,
#     normalized_ds2_bis,
#     normalized_ds1_bis,
# ]


# get the model skeleton
# tf.reset_default_graph()
# loader = tf.train.import_meta_graph(l_ckpts[-1] + '.meta', clear_devices=True)

###########################
#
# single core
#
###########################

# # init
# losses = {}
# acces = {}
# loss = np.zeros(xm.shape).ravel()
# acc = np.zeros(xm.shape).ravel()
#
# # start looping
# for _step, _ckpt, _state in tqdm(zip(l_steps, l_ckpts, l_states), desc='check point'):
#     with tf.Session(config=config) as sess:
#         sess.run([tf.local_variables_initializer()])
#         loader.restore(sess, _ckpt)
#         for i in tqdm(range(loss.size), desc='loss length'):
#             graph = tf.get_default_graph()
#             # print_nodes_name_shape(graph)
#             loss[i], acc[i] = feed_forward(sess=sess,
#                                            graph=graph,
#                                            state=_state,
#                                            direction_2D=[normalized_ds1, normalized_ds1_bis],
#                                            xcoord=xcoord[i // x_nb],  # chunk
#                                            ycoord=ycoord[i % y_nb],  # remainder
#                                            inputs=np.random.randn(8, 20, 20, 1) * 3 + np.ones((8, 20, 20, 1)),  # new random inputs
#                                            outputs=outputs
#                                            )
#
#     losses[_step] = loss.reshape(xm.shape)
#     acces[_step] = acc.reshape(xm.shape)
#
#     # plot surface
#     fig, (ax1, ax2) = plt.subplots(122)
#     cs1 = ax1.contour(xm, ym, loss)
#     plt.clabel(cs1, inline=1, fontsize=10)
#     cs2 = ax2.contour(xm, ym, acc)
#     plt.clabel(cs2, inline=1, fontsize=10)
#     plt.show()
#
#     pd.DataFrame(losses[_step]).to_csv('./dummy/lss_step{}.csv'.format(_step))
#     pd.DataFrame(acces[_step]).to_csv('./dummy/acc_step{}.csv'.format(_step))
#     csv_interp(xm, ym, losses[_step], './dummy/paraview_lss_step{}'.format(_step))
#     csv_interp(xm, ym, acces[_step], './dummy/paraview_lss_step{}'.format(_step))

#################################################
#
#  use multiprocessing module to accelerate (failed)
#
#################################################
# https://jonasteuwen.github.io/numpy/python/multiprocessing/2017/01/07/multiprocessing-numpy-array.html
# from itertools import repeat
# from multiprocessing.sharedctypes import RawArray
# import multiprocessing as mp
#
# def feed_forward_MP(ckpt_path, state, direction_2D, x_mesh, y_mesh, i, block_size):
#     import tensorflow as tf
#     print(mp.current_process())
#     assert isinstance(direction_2D, list)
#     assert isinstance(x_mesh, np.ndarray)
#     assert isinstance(y_mesh, np.ndarray)
#     try:
#         xcoords = x_mesh[i * block_size: (i + 1) * block_size]
#         ycoords = y_mesh[i * block_size: (i + 1) * block_size]
#     except:
#         xcoords = x_mesh[-(x_mesh % block_size):]
#         ycoords = y_mesh[-(y_mesh % block_size):]
#     loader = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
#     tmp_lss = np.ctypeslib.as_array(shared_loss)
#     tmp_acc = np.ctypeslib.as_array(shared_acc)
#
#     with tf.Session() as sess:
#         sess.run(tf.local_variables_initializer())
#         print('hello!')
#         loader.restore(sess, ckpt_path)
#         graph = tf.get_default_graph()
#         # get tensors and ops
#         loss_tensor = graph.get_tensor_by_name('metrics/loss/value:0')
#         acc_tensor = graph.get_tensor_by_name('metrics/acc/value:0')
#         new_loss_update_op = graph.get_operation_by_name('metrics/loss/update_op')
#         new_acc_update_op = graph.get_operation_by_name('metrics/acc/update_op')
#         new_input_ph = graph.get_tensor_by_name('input_ph:0')
#         new_output_ph = graph.get_tensor_by_name('output_ph:0')
#         new_BN_ph = graph.get_tensor_by_name('BN_phase:0')
#
#         for i in tqdm(range(len(xcoords))):
#             print(i)
#             # apply changes
#             dx = {k: xcoords[i] * v for k, v in direction_2D[0].items()}  # step size * direction x
#             dy = {k: ycoords[i] * v for k, v in direction_2D[1].items()}  # step size * direction y
#             change = {k: _dx + _dy for (k, _dx), (_, _dy) in zip(dx.items(), dy.items())}
#             for k, v in state.items():
#                 move_state(sess, name=k, leap=change[k])
#
#             _loss, _acc, _, _ = sess.run([loss_tensor, acc_tensor, new_acc_update_op, new_loss_update_op],
#                                        feed_dict={new_input_ph: inputs,
#                                                   new_output_ph: outputs,
#                                                   new_BN_ph: False,
#                                                   })
#             tmp_lss[xcoords[i], ycoords[i]] = _loss
#             tmp_acc[xcoords[i], ycoords[i]] = _acc
#
#
#
# # init
# block_size = 100
# nb = xm.size // block_size
# remainder = xm.size % block_size
# rlt_loss = np.ctypeslib.as_ctypes(np.zeros(xm.shape))
# rlt_acc = np.ctypeslib.as_ctypes(np.zeros(xm.shape))
# shared_loss = RawArray(rlt_loss._type_, rlt_loss)
# shared_acc = RawArray(rlt_acc._type_, rlt_acc)
#
# p = mp.Pool()
# for _step, _ckpt, _state, _dir, _dirbis in tqdm(zip(l_steps, l_ckpts, l_states, l_directions, l_directions_bis), desc='check point'):
#     if remainder != 0:
#         args = [(_c, _s, _d, _xm, _ym, i, _bl_s)
#                 for _c, _s, _d, _xm, _ym, i, _bl_s in tqdm(zip(repeat(_ckpt),
#                                                                repeat(_state),
#                                                                repeat([_dir, _dirbis]),
#                                                                repeat(xm),
#                                                                repeat(ym),
#                                                                range(nb + 1),
#                                                                repeat(block_size)
#                                                                ), desc='step')
#                 ]  # todo: implement MP progress bar
#     else:
#         args = [(_c, _s, _d, _xm, _ym, i, _bl_s)
#                 for _c, _s, _d, _xm, _ym, i, _bl_s in tqdm(zip(repeat(_ckpt),
#                                                                repeat(_state),
#                                                                repeat([_dir, _dirbis]),
#                                                                repeat(xm),
#                                                                repeat(ym),
#                                                                range(nb),
#                                                                repeat(block_size)
#                                                                ), desc='step')
#                 ]  # todo: implement MP progress bar
#     print('\n last batch')
#
#
#     p.starmap(feed_forward_MP, args)
#     rlt_loss = np.ctypeslib.as_array(shared_loss)
#     rlt_acc = np.ctypeslib.as_array(shared_acc)
#     pd.DataFrame(rlt_loss).to_csv('./dummy/lss_step{}.csv'.format(_step))
#     pd.DataFrame(rlt_acc).to_csv('./dummy/acc_step{}.csv'.format(_step))
#     csv_interp(xm, ym, np.log(rlt_loss), './dummy/paraview_lss_step{}'.format(_step))
#     csv_interp(xm, ym, np.log(rlt_acc), './dummy/paraview_lss_step{}'.format(_step))


#################################################
#
#  use mpi4py to accelerate
#
#################################################
# from mpi4py import MPI
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import numpy as np
# from tqdm import tqdm
# from util import print_nodes_name_shape
# import copy
# import pandas as pd
# from scipy.interpolate import interp2d
# from scipy.spatial.distance import cosine, euclidean
# import os
# # prevent GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#
# #note: state includes weights, bias, BN
# def get_diff_state(state1, state2):
#     assert isinstance(state1, dict)
#     assert isinstance(state2, dict)
#     return {k: v2 - v1 for (k, _), (v1, v2) in zip(state1.items(), state2.items())}
#
#
# def get_random_state(state):
#     assert isinstance(state, dict)
#     return {k: np.random.randn(*v.shape) for k, v in state.items()}
#
#
# def get_state(ckpt_path):
#     tf.reset_default_graph()
#     loader = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
#     state = {}
#     with tf.Session() as sess:
#         loader.restore(sess, ckpt_path)
#         # print_nodes_name_shape(tf.get_default_graph())
#         ckpt_weight_names = []
#         for node in tf.get_default_graph().as_graph_def().node:
#             if node.name.endswith('w1') or \
#                     node.name.endswith('b1') or \
#                     node.name.endswith('beta') or \
#                     node.name.endswith('gamma'):
#                 ckpt_weight_names.append(node.name + ':0')
#
#         # get weights/bias
#         for k in ckpt_weight_names:
#             v = sess.run(k)
#             state[k] = v
#     return state
#
#
# def _normalize_direction(perturbation, weight):
#     assert isinstance(perturbation, np.ndarray)
#     assert isinstance(weight, np.ndarray)
#     norm_w = np.linalg.norm(weight)
#     norm_pert = np.linalg.norm(perturbation)
#     print('norm_w, norm_pert: {}, {}'.format(norm_w, norm_pert))
#     perturbation *= norm_w / (norm_pert)
#     return perturbation, weight
#
#
# def normalize_state(directions, state):
#     assert isinstance(directions, dict)
#     assert isinstance(state, dict)
#     backup = copy.deepcopy(directions)
#     for (d_name, direct), (name, weight) in zip(backup.items(), state.items()):
#         _normalize_direction(direct, weight)
#     return backup
#
#
# def move_state(sess, name, value, leap):
#     # if ('beta' not in name) and ('gamma' not in name):
#     weight = sess.graph.get_tensor_by_name(name)
#     try:
#         # print('leap: {}'.format(leap))
#         # initial = sess.run(weight)
#         # print('\n{} init: {}'.format(name, initial))
#         # assign = sess.run(tf.assign(weight, value))
#         # print('\n{} Assigned: {}'.format(name, assign))
#         after = sess.run(tf.assign(weight, value + leap))
#         # print('\nAfter: {}'.format(after))
#     except Exception as e:
#         print('initial shape: \n', sess.run(value).shape)
#         print('leap shape: \n', leap.shape)
#         print('\nError threw while trying to move weight: {}'.format(name))
#         print(e)
#
#
# def feed_forward(sess, graph, state, direction_2D, xcoord, ycoord, inputs, outputs, comm=None):
#     '''return loss and acc'''
#     assert isinstance(direction_2D, list), 'dir should be list'
#     assert isinstance(xcoord, float), 'xcoord should be float'
#     assert isinstance(ycoord, float), 'ycoord should be float'
#     # print('inputs: {}'.format(inputs))
#     # print('outputs: {}'.format(outputs))
#     # print('xcoord:', xcoord)
#     sess.run([tf.local_variables_initializer()])  # note: should initialize here otherwise it will keep cumulating for the average
#     # print(sess.run(tf.local_variables()))  #note: uncomment this line to see if local_variables(metrics/loss/count...) are well init
#     # change state in the neural network
#     new_logits = graph.get_tensor_by_name('model/MLP/logits:0')
#     loss_tensor = graph.get_tensor_by_name('metrics/loss/value:0')
#     acc_tensor = graph.get_tensor_by_name('metrics/acc/value:0')
#     new_loss_update_op = graph.get_operation_by_name('metrics/loss/update_op')
#     new_acc_update_op = graph.get_operation_by_name('metrics/acc/update_op')
#     new_input_ph = graph.get_tensor_by_name('input_ph:0')
#     new_output_ph = graph.get_tensor_by_name('output_ph:0')
#     new_BN_ph = graph.get_tensor_by_name('BN_phase:0')
#
#     # print('inputs avg: {}, outputs avg: {}'.format(np.mean(inputs), np.mean(outputs)))
#
#     dx = {k: xcoord * v for k, v in direction_2D[0].items()}  # step size * direction x
#     dy = {k: ycoord * v for k, v in direction_2D[1].items()}  # step size * direction y
#     change = {k: _dx + _dy for (k, _dx), (_, _dy) in zip(dx.items(), dy.items())}
#     # calculate the perturbation
#     for k, v in state.items():
#         move_state(sess, name=k, value=v, leap=change[k])
#
#     # feed forward with batches
#     for repeat in range(2):
#         #note: (TF intrisic: at least 2 times, or loss/acc 0.0)should iterate at least several time, Or loss=acc=0, since there's a counter for the average
#         new_log, loss_ff, acc_ff, _, _ = sess.run([new_logits, loss_tensor, acc_tensor, new_acc_update_op, new_loss_update_op],
#                                    feed_dict={new_input_ph: inputs,
#                                               new_output_ph: outputs,
#                                               new_BN_ph: True,
#                                               #note: (TF1.14)WTF? here should be True while producing loss-landscape
#                                               #fixme: should check if the mov_avg/mov_std/beta/gamma change
#                                               })
#     if comm is not None:
#         if comm.Get_rank() != 0:
#             comm.send(1, dest=0, tag=tag_compute)
#
#     # print('lss:{}, acc:{}, predict:{}'.format(loss, acc, np.mean(new_log)))
#     return loss_ff, acc_ff
#
#
# def csv_interp(x_mesh, y_mesh, metrics_tensor, out_path, interp_scope=5):
#     new_xmesh = np.linspace(np.min(x_mesh), np.max(x_mesh), interp_scope * x_mesh.shape[0])
#     new_ymesh = np.linspace(np.min(y_mesh), np.max(y_mesh), interp_scope * x_mesh.shape[1])
#     newxx, newyy = np.meshgrid(new_xmesh, new_ymesh)
#
#     # interpolation
#     interpolation = interp2d(x_mesh, y_mesh, metrics_tensor, kind='cubic')
#     zval = interpolation(new_xmesh, new_ymesh)
#     pd.DataFrame({'xcoord': newxx.ravel(),
#                   'ycoord': newyy.ravel(),
#                   'zval': zval.ravel()}
#                  ).to_csv(out_path, index=False)
#
#
# config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 1})
#
# # mac OS
# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# # prevent GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
#
#
# def clean(array):
#     assert isinstance(array, np.ndarray)
#     array[np.where(array == np.nan)] = 1e-9
#     # array[np.where(array == 0)] = 1e-9
#     array[np.where(array == np.inf)] = 1e9
#     array[np.where(array == -np.inf)] = -1e9
#     return array
#
#
# def feed_forward_MP(ckpt_path, state, direction_2D, x_mesh, y_mesh, comm=None):
#     assert isinstance(direction_2D, list)  # list of dicts
#     assert isinstance(x_mesh, np.ndarray) and x_mesh.ndim == 1
#     assert isinstance(y_mesh, np.ndarray) and y_mesh.ndim == 1
#
#     loader = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
#     tmp_loss, tmp_acc = np.zeros(x_mesh.size), np.zeros(x_mesh.size)
#
#     with tf.Session() as sess:
#         sess.run([tf.local_variables_initializer()])
#         # print(sess.run(tf.local_variables()))  #note: local_variables include metrics/acc/total; metrics/loss/count; metrics/acc/count...
#         loader.restore(sess, ckpt_path)
#         graph = tf.get_default_graph()
#         # print_nodes_name_shape(graph)
#         for i in range(x_mesh.size):
#             inputs = np.random.randn(8, 20, 20, 1) + np.ones((8, 20, 20, 1)) * 8  # noise + avg:8
#             outputs = np.ones((8, 20, 20, 1)) * 64   # avg: 9
#             tmp_loss[i], tmp_acc[i] = feed_forward(sess=sess,
#                                            graph=graph,
#                                            state=state,
#                                            direction_2D=direction_2D,
#                                            xcoord=float(x_mesh[i]),   #note: can debug with 0.0
#                                            ycoord=float(y_mesh[i]),   #note: can debug with 0.0
#                                            inputs=inputs,
#                                            outputs=outputs,
#                                            comm=comm)
#
#     # print('\nlss', tmp_loss)
#     # print('\nacc', tmp_acc)
#     return tmp_loss.astype(np.float32), tmp_acc.astype(np.float32)
#
# l_steps = [str(i * 1000) for i in range(10)]
# l_ckpts = ['./dummy/ckpt/step{}'.format(i) for i in l_steps]
# l_states = [get_state(_c) for _c in l_ckpts]
# l_directions = [normalize_state(get_random_state(_s), _s) for _s in l_states]
# l_directions_bis = [normalize_state(get_random_state(_s), _s) for _s in l_states]
#
# # compute l2-norm and cos
# l_angles = []
# for i, (dict1, dict2) in enumerate(zip(l_directions, l_directions_bis)):
#     tmp = {}
#     for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
#         assert k1 == k2, 'Found different weights names'
#         tmp[k1] = cosine(v1, v2)
#     l_angles.append(tmp)
#
# l_L2norm = []
# for i, (dict1, dict2) in enumerate(zip(l_directions, l_directions_bis)):
#     tmp = {}
#     for (k1, v1), (k2, v2) in zip(dict1.items(), dict2.items()):
#         assert k1 == k2, 'Found different weights names'
#         tmp[k1] = euclidean(v1, v2)
#     l_angles.append(tmp)
#
# # write cos and l2-norm to xlsw
# for i, angle in enumerate(l_angles):
#     with pd.ExcelWriter('./dummy/cosin.xlsx', engine='xlsxwriter') as writer:
#         angle['step{}'.format(i * 1000)].to_excel(writer, index=False, header=False)
#
# # create direction and surface .h5
# x_min, x_max, x_nb = -1, 1, 51
# y_min, y_max, y_nb = -1, 1, 51
# xcoord = np.linspace(x_min, x_max, x_nb).ravel()
# ycoord = np.linspace(y_min, y_max, y_nb).ravel()
# xm, ym = np.meshgrid(xcoord, ycoord)
# xm = xm.astype(np.float32)
# ym = ym.astype(np.float32)
#
# # get the model skeleton
# # tf.reset_default_graph()
# # loader = tf.train.import_meta_graph(ckpt1_path + '.meta', clear_devices=True)
#
# # calculate loss/acc for each point on the surface (try first with only for loop)
# # start feeding
#
#
# communicator = MPI.COMM_WORLD
# rank = communicator.Get_rank()
# nb_process = communicator.Get_size()
#
# total_computation = xm.size
# remainder = total_computation % (nb_process - 1)  # master node manage remainder
# bus_per_rank = total_computation // (nb_process - 1)  # sub-nodes compute others
#
# print('MPI_version', MPI.get_vendor())
# print('This rank is:', rank)
# print('nb_process', nb_process)
# # **************************************************************************************************** I'm a Barrier
# communicator.Barrier()
#
# #note: numpy use Send/Recv, list/dict use send/recv
# tag_compute = 0
# tag_end = 99
#
# for _step, _ckpt, _state, _dir, _dir_bis in tqdm(zip(l_steps, l_ckpts, l_states, l_directions, l_directions_bis)
#                                                  , desc='Checkpoint'):
#     # **************************************************************************************************** I'm a Barrier
#     communicator.Barrier()
#     # fixme: put it somewhere else
#
#     # init placeholder
#     if rank == 0:
#         shared_lss = np.empty(xm.shape, dtype=np.float32).ravel()
#         shared_acc = np.empty(xm.shape, dtype=np.float32).ravel()
#         loss_ph = np.empty(bus_per_rank, dtype=np.float32)
#         acc_ph = np.empty(bus_per_rank, dtype=np.float32)
#         _dir1_ph = None
#         _dir2_ph = None
#         xm_ph = None
#         ym_ph = None
#
#         pbar = tqdm(total=total_computation)
#         update_msg = None
#
#
#
#     else:
#         shared_lss = None
#         shared_acc = None
#         loss_ph = None
#         acc_ph = None
#         xm_ph = np.empty(bus_per_rank, dtype=np.float32)
#         ym_ph = np.empty(bus_per_rank, dtype=np.float32)
#         update_msg = None
#
#
#     # **************************************************************************************************** I'm a Barrier
#     communicator.Barrier()
#
#
#     # from 0 send buses to sub-process
#     if rank == 0:
#         print('\n****Start scattering')
#         try:
#             # send order
#             count = 0
#             remaining = nb_process - 1
#             for _rank in tqdm(range(1, nb_process)):
#                 communicator.send(_dir, dest=_rank, tag=31)
#                 communicator.send(_dir_bis, dest=_rank, tag=32)
#                 communicator.Send(xm.ravel()[(_rank - 1) * bus_per_rank: _rank * bus_per_rank], dest=_rank, tag=44)
#                 communicator.Send(ym.ravel()[(_rank - 1) * bus_per_rank: _rank * bus_per_rank], dest=_rank, tag=55)
#                 count += 1
#
#             print('Rank {} sent successfully'.format(rank))
#
#
#         except Exception as e:
#             print('While sending buses, \nRank {} throws error: {}'.format(rank, e))
#             break
#
#         try:
#             _loss, _acc = feed_forward_MP(ckpt_path=_ckpt,
#                                           state=_state,
#                                           direction_2D=[_dir, _dir_bis],
#                                           x_mesh=xm.ravel()[-remainder:],
#                                           y_mesh=ym.ravel()[-remainder:],
#                                           comm=communicator
#                                           )
#
#             shared_lss[-remainder:] = _loss
#             shared_acc[-remainder:] = _acc
#
#             while remaining > 0:
#                 s = MPI.Status()
#                 communicator.Probe(status=s)
#                 if s.tag == tag_compute:
#                     update_msg = communicator.recv(tag=tag_compute)
#                     pbar.update(1)
#                 elif s.tag == tag_end:
#                     update_msg = communicator.recv(tag=tag_end)
#                     remaining -= 1
#                     print('remaining: {}', remaining)
#
#             print('Rank 0 out of while loop')
#
#         except Exception as e:
#             print('While computing, \nRank {} throws error: {}'.format(rank, e))
#
#     else:
#         try:
#             # receive
#             _dir1_ph = communicator.recv(source=0, tag=31)
#             _dir2_ph = communicator.recv(source=0, tag=32)
#             communicator.Recv(xm_ph, source=0, tag=44)
#             communicator.Recv(ym_ph, source=0, tag=55)
#             print('Rank {} received successfully'.format(rank))
#         except Exception as e:
#             print('While sending buses, \nRank {} throws error: {}'.format(rank, e))
#             break
#
#         try:
#             # compute
#             _loss, _acc = feed_forward_MP(ckpt_path=_ckpt,
#                                           state=_state,
#                                           direction_2D=[_dir1_ph, _dir2_ph],
#                                           x_mesh=xm_ph,
#                                           y_mesh=ym_ph,
#                                           comm=communicator
#                                           )
#             communicator.send(1, dest=0, tag=tag_end)
#         except Exception as e:
#             print('While computing, \nRank {} throws error: {}'.format(rank, e))
#
#
#     # **************************************************************************************************** I'm a Barrier
#     print('Hello!')
#     communicator.Barrier()
#     print('\n****Start gathering')
#
#     # Send back and Gathering
#     if rank == 0:
#         try:
#             # gathering
#             for _rank in tqdm(range(1, nb_process)):
#                 communicator.Recv(loss_ph, source=_rank, tag=91)
#                 communicator.Recv(acc_ph, source=_rank, tag=92)
#                 print('Received from rank {} successfully'.format(_rank))
#                 shared_lss[(_rank - 1) * bus_per_rank: _rank * bus_per_rank] = loss_ph
#                 shared_acc[(_rank - 1) * bus_per_rank: _rank * bus_per_rank] = acc_ph
#         except Exception as e:
#             print('While gathering buses, \nRank {} throws error: {}'.format(rank, e))
#
#     else:
#         try:
#             # send back
#             communicator.Send(_loss, dest=0, tag=91)
#             communicator.Send(_acc, dest=0, tag=92)
#             print('Rank {} sent successfully'.format(rank))
#         except Exception as e:
#             print('While gathering buses, \nRank {} throws error: {}'.format(rank, e))
#
#     # **************************************************************************************************** I'm a Barrier
#     communicator.Barrier()
#     # save and plot
#     if rank == 0:
#         shared_lss = shared_lss.reshape(xm.shape)
#         shared_acc = shared_acc.reshape(xm.shape)
#
#         # take the log for loss
#         shared_lss = clean(shared_lss)
#         # shared_lss = np.log(shared_lss)
#
#         # plot results
#         # fig, ax1 = plt.subplots(1)
#         # cs1 = ax1.contour(xm, ym, shared_lss)
#         # plt.clabel(cs1, inline=1, fontsize=10)
#         # fig2, ax2 = plt.subplots(1)
#         # cs2 = ax2.contour(xm, ym, shared_acc)
#         # plt.clabel(cs2, inline=1, fontsize=10)
#         # plt.show()
#
#         pd.DataFrame(shared_lss).to_csv('./dummy/lss_step{}.csv'.format(_step), index=False)
#         pd.DataFrame(shared_acc).to_csv('./dummy/acc_step{}.csv'.format(_step), index=False)
#         # csv_interp(xm, ym, shared_lss, './dummy/paraview_lss_step{}.csv'.format(_step))
#         # csv_interp(xm, ym, shared_acc, './dummy/paraview_lss_step{}.csv'.format(_step))
#
#     # **************************************************************************************************** I'm a Barrier
#     communicator.Barrier()


####################################################################
#
#                  Test batch norm is_training param
#
####################################################################

# import tensorflow as tf
# from tqdm import tqdm
# import numpy as np
# from util import print_nodes_name_shape, check_N_mkdir
# from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
# from tensorflow.python.framework import dtypes
# import os
#
#
# input_ph = tf.placeholder(tf.float32, shape=(None, 2, 2, 1), name='input_ph')
# output_ph = tf.placeholder(tf.float32, shape=(None, 2, 2, 3), name='output_ph')
# is_training = tf.placeholder(tf.bool, shape=[], name='is_training')
#
# # build a one layer Full layer with BN and save 2 ckpts
# with tf.name_scope('model'):
#     out = tf.reshape(input_ph, shape=(-1, 2 * 2 * 1), name='flatten')
#     with tf.variable_scope('dnn1', reuse=False):
#         w1 = tf.get_variable('w1', dtype=tf.float32, shape=[4 * 1, 4 * 3], initializer=tf.initializers.glorot_normal())
#         # b1 = tf.get_variable('b1', dtype=tf.float32, shape=[4 * 3], initializer=tf.initializers.glorot_normal())
#     # out = tf.matmul(out, w1) + b1
#     out = tf.matmul(out, w1)
#     out = tf.layers.batch_normalization(out, training=is_training, name='BN')
#     logits = tf.nn.relu(out)
#     logits = tf.reshape(logits, shape=(-1, 2, 2, 3))
#
# with tf.name_scope('loss'):
#     MSE = tf.losses.mean_squared_error(labels=output_ph, predictions=logits)
#
# with tf.name_scope('operations'):
#     opt = tf.train.AdamOptimizer(learning_rate=0.0001, name='Adam')
#     grads = opt.compute_gradients(MSE)
#     train_op = opt.apply_gradients(grads, name='apply_grad')
#
# # train
# with tf.Session() as sess:
#     # prepare
#     graph = tf.get_default_graph()
#     print_nodes_name_shape(graph)
#     saver = tf.train.Saver()
#
#     # init variables
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#     print(update_ops)  # note: [<tf.Operation 'model/BN/cond_2/Merge' type=Merge>, <tf.Operation 'model/BN/cond_3/Merge' type=Merge>]
#     saver.save(sess, './dummy/ckpt/step0')
#     # train
#     print('\n*******************************************************************************************First training')
#
#     for i in tqdm(range(100)):
#         inputs = np.ones((8, 2, 2, 1)) + np.random.randn(8, 2, 2, 1)
#         outputs = np.arange(8 * 2 * 2 * 3).reshape((8, 2, 2, 3))
#         with tf.variable_scope('', reuse=True):
#             mov_avg, mov_std, beta, gamma = sess.run([tf.get_variable('BN/moving_mean'),
#                                          tf.get_variable('BN/moving_variance'),
#                                                       tf.get_variable('BN/beta'),
#                                                       tf.get_variable('BN/gamma')])
#         if i == 0 or i == 99:
#             print('\nmov_avg: {}, \nmov_std: {}, \nbeta: {}, \ngamma: {}'.format(mov_avg, mov_std, beta, gamma))
#         _, _ = sess.run([train_op, tf.get_collection(tf.GraphKeys.UPDATE_OPS)], feed_dict={
#             input_ph: inputs,
#             output_ph: outputs,
#             is_training: True,
#         })
#
#     print('\n*******************************************************************************************First testing')
#     for i in tqdm(range(100)):
#         inputs = np.ones((8, 2, 2, 1)) + np.random.randn(8, 2, 2, 1)
#         outputs = np.arange(8 * 2 * 2 * 3).reshape((8, 2, 2, 3))
#         with tf.variable_scope('', reuse=True):
#             mov_avg, mov_std, beta, gamma = sess.run([tf.get_variable('BN/moving_mean'),
#                                          tf.get_variable('BN/moving_variance'),
#                                                       tf.get_variable('BN/beta'),
#                                                       tf.get_variable('BN/gamma')])
#         if i == 0 or i == 99:
#             print('\nmov_avg: {}, \nmov_std: {}, \nbeta: {}, \ngamma: {}'.format(mov_avg, mov_std, beta, gamma))
#         _ = sess.run([graph.get_tensor_by_name('model/Reshape:0')], feed_dict={
#             input_ph: inputs,
#             is_training: False,
#         })
#
#     print('\n*******************************************************************************************Second training')
#     for i in tqdm(range(100)):
#         inputs = np.ones((8, 2, 2, 1)) + np.random.randn(8, 2, 2, 1)
#         outputs = np.arange(8 * 2 * 2 * 3).reshape((8, 2, 2, 3))
#         with tf.variable_scope('', reuse=True):
#             mov_avg, mov_std, beta, gamma = sess.run([tf.get_variable('BN/moving_mean'),
#                                                       tf.get_variable('BN/moving_variance'),
#                                                       tf.get_variable('BN/beta'),
#                                                       tf.get_variable('BN/gamma')])
#         if i == 0 or i == 99:
#             print('\nmov_avg: {}, \nmov_std: {}, \nbeta: {}, \ngamma: {}'.format(mov_avg, mov_std, beta, gamma))
#         _, _ = sess.run([train_op, tf.get_collection(tf.GraphKeys.UPDATE_OPS)], feed_dict={
#             input_ph: inputs,
#             output_ph: outputs,
#             is_training: True,
#         })
#
#     print('\n*******************************************************************************************Second testing')
#     for i in tqdm(range(100)):
#         inputs = np.ones((8, 2, 2, 1)) + np.random.randn(8, 2, 2, 1)
#         outputs = np.arange(8 * 2 * 2 * 3).reshape((8, 2, 2, 3))
#         with tf.variable_scope('', reuse=True):
#             mov_avg, mov_std, beta, gamma = sess.run([tf.get_variable('BN/moving_mean'),
#                                                       tf.get_variable('BN/moving_variance'),
#                                                       tf.get_variable('BN/beta'),
#                                                       tf.get_variable('BN/gamma')])
#         if i == 0 or i == 99:
#             print('\nmov_avg: {}, \nmov_std: {}, \nbeta: {}, \ngamma: {}'.format(mov_avg, mov_std, beta, gamma))
#         _ = sess.run([graph.get_tensor_by_name('model/Reshape:0')], feed_dict={
#             input_ph: inputs,
#             is_training: False,
#         })
#
#         # print moving avg/std
#     saver.save(sess, './dummy/ckpt/step100')
#
#
# def freeze_ckpt_for_inference(ckpt_path=None, conserve_nodes=None):
#     # clean graph first
#     tf.reset_default_graph()
#     # freeze ckpt then convert to pb
#     new_input = tf.placeholder(tf.float32, shape=[None, 10, 10, 1], name='new_input')
#     # new_is_training = tf.placeholder_with_default(True, [], name='new_is_training')
#     new_is_training = tf.placeholder(tf.bool, [], name='new_is_training')
#     restorer = tf.train.import_meta_graph(
#         ckpt_path + '.meta',
#         input_map={
#             'input_ph': new_input,
#             'is_training': new_is_training,
#         },
#         clear_devices=True,
#     )
#
#     input_graph_def = tf.get_default_graph().as_graph_def()
#     check_N_mkdir('./dummy/pb/')
#     check_N_mkdir('./dummy/tb/')
#
#     # freeze to pb
#     with tf.Session() as sess:
#         # restore variables
#         restorer.restore(sess, './dummy/ckpt/step100')
#         # convert variable to constant
#         output_graph_def = tf.graph_util.convert_variables_to_constants(
#             sess=sess,
#             input_graph_def=input_graph_def,
#             output_node_names=conserve_nodes,
#         )
#
#         # save to pb
#         with tf.gfile.GFile('./dummy/pb/freeze.pb', 'wb') as f:  # 'wb' stands for write binary
#             f.write(output_graph_def.SerializeToString())
#
#
# def optimize_graph_for_inference(pb_dir=None, conserve_nodes=None):
#     tf.reset_default_graph()
#     check_N_mkdir(pb_dir)
#
#     # import pb file
#     with tf.gfile.FastGFile(pb_dir + 'freeze.pb', "rb") as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#
#     # optimize graph
#     optimize_graph_def = optimize_for_inference(input_graph_def=graph_def,
#                                                 input_node_names=['new_input',
#                                                                   'new_is_training'  #note:
#                                                                   ],
#                                                 output_node_names=conserve_nodes,
#                                                 placeholder_type_enum=[dtypes.float32.as_datatype_enum,
#                                                                        dtypes.bool.as_datatype_enum,
#                                                                        dtypes.float32.as_datatype_enum,
#                                                                        ]
#                            )
#     with tf.gfile.GFile(pb_dir + 'optimize.pb', 'wb') as f:
#         f.write(optimize_graph_def.SerializeToString())
#
# conserve_nodes = ['model/Reshape']
# freeze_ckpt_for_inference(ckpt_path='./dummy/ckpt/step100', conserve_nodes=conserve_nodes)
# optimize_graph_for_inference(pb_dir='./dummy/pb/', conserve_nodes=conserve_nodes)
#
# # cleaning
# tf.reset_default_graph()
#
# # load pb
# with tf.gfile.FastGFile('./dummy/pb/optimize.pb', "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())
#
# print('\n ***********************************************************************************************Now inference')
# # inference
# with tf.Session() as sess:
#     tf.graph_util.import_graph_def(
#         graph_def,
#     )
#     # prepare
#     G = tf.get_default_graph()
#     print_nodes_name_shape(G)
#     new_input = G.get_tensor_by_name('import/new_input:0')
#     new_is_training = G.get_tensor_by_name('import/new_is_training:0')
#     new_output = G.get_tensor_by_name('import/' + conserve_nodes[-1] + ':0')
#
#     # train
#     for i in tqdm(range(100)):
#         inputs = np.ones((8, 2, 2, 1)) + np.random.randn(8, 2, 2, 1)
#         outputs = np.arange(8 * 2 * 2 * 3).reshape((8, 2, 2, 3))
#         # print moving avg/std
#         with tf.variable_scope('', reuse=True):
#             mov_avg, mov_std, beta, gamma = sess.run([G.get_tensor_by_name('import/BN/moving_mean:0'),
#                                                       G.get_tensor_by_name('import/BN/moving_variance:0'),
#                                                       G.get_tensor_by_name('import/BN/beta:0'),
#                                                       G.get_tensor_by_name('import/BN/gamma:0')])
#         if i == 0 or i == 99:
#             print('\nmov_avg: {}, \nmov_std: {}, \nbeta: {}, \ngamma: {}'.format(mov_avg, mov_std, beta, gamma))
#         new_out = sess.run([new_output], feed_dict={
#             new_input: inputs,
#             new_output: outputs,
#             new_is_training: False,
#         })
#         # print('out: {}'.format(new_out))

# import sys, time
#
# from PyQt5.QtWidgets import *
# from PyQt5.QtGui import *
# from PyQt5.QtCore import *
#
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.figure import Figure
#
# import matplotlib.pyplot as plt
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors
# import matplotlib.colorbar as mcolorbar
#
# import numpy as np
# import pylab as pl
#
# import random
#
#
# class Example(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.initUI()
#
#     def initUI(self):
#         self.central_widget = QWidget()
#         self.setCentralWidget(self.central_widget)
#         self.editLayout = QWidget()
#         self.edit = QLineEdit('chfghfghf', self)
#         self.edit.setDragEnabled(True)
#
#         self.left_layout = QVBoxLayout()
#         self.left_layout.addWidget(self.edit)
#
#         self.editLayout.setLayout(self.left_layout)
#
#         #Create the right layout that contains the plot canvas.
#         self.plotLayout = QWidget();
#
#         canvas = Create_Canvas(self)
#
#         self.button = QPushButton('Plot')
#
#         # set the layout
#         self.right_layout = QVBoxLayout()
#         self.right_layout.addWidget(canvas)
#         self.right_layout.addWidget(self.button)
#
#         self.plotLayout.setLayout(self.right_layout)
#
#         splitter_filebrowser = QSplitter(Qt.Horizontal)
#         splitter_filebrowser.addWidget(self.editLayout)
#         splitter_filebrowser.addWidget(self.plotLayout)
#         splitter_filebrowser.setStretchFactor(1, 1)
#
#         hbox = QHBoxLayout(self)
#         hbox.addWidget(splitter_filebrowser)
#
#         self.centralWidget().setLayout(hbox)
#
#         self.setWindowTitle('Simple drag & drop')
#         self.setGeometry(750, 100, 600, 500)
#
#
# class Create_Canvas(QWidget):
#     def __init__(self, parent):
#         QWidget.__init__(self, parent)
#         self.setAcceptDrops(True)
#
#         self.figure = plt.figure()
#         self.canvas = FigureCanvas(self.figure)
#         toolbar = NavigationToolbar(self.canvas, self)
#
#         self.right_layout = QVBoxLayout()
#         self.right_layout.addWidget(self.canvas)
#         self.right_layout.addWidget(toolbar)
#         # set the layout of this widget, otherwise the elements will not be seen.
#         self.setLayout(self.right_layout)
#         # plot some stuff
#         self.ax = self.figure.add_subplot(111)
#         self.ax.plot([1,2,5])
#         # finally draw the canvas
#         self.canvas.draw()
#
#     def dragEnterEvent(self, e):
#         print('entering')
#         if e.mimeData().hasFormat('text/plain'):
#             e.accept()
#         else:
#             e.ignore()
#
#     def dragMoveEvent(self, e):
#         print('drag moving')
#
#     def dropEvent(self, e):
#         print("dropped")
#
#
# if __name__ == '__main__':
#
#     app = QApplication(sys.argv)
#     ex = Example()
#     ex.show()
#     app.exec_()