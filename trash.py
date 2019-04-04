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
import tensorflow as tf
import numpy as np
import os
import h5py
import multiprocessing as mp
from itertools import repeat
#
# def write_h5(args):
#     x, is_training = args
#     with h5py.File('./{}_{}.h5'.format('train' if is_training else 'test', x), 'w') as f:
#         h = w = np.arange(-1, 1, 0.02)
#         hh, _ = np.meshgrid(h, w)
#         a = hh ** 2
#         b = np.add(a + 1, np.random.randn(100, 100))  #do something and add gaussian noise
#         f.create_dataset('X', shape=(100, 100), dtype='float32', data=a)
#         f.create_dataset('y', shape=(100, 100), dtype='float32', data=b)
#
#
# def input_pipeline(window_size, batch_size, is_train=True, ncores=mp.cpu_count()):
#     flist = []
#     for dirpath, _, fnames in os.walk('./'):
#         for fname in fnames:
#             if fname.startswith('train' if is_train else 'test') and fname.endswith('.h5'):
#                 print(fname)
#                 flist.append((os.path.abspath(os.path.join(dirpath, fname)), str(window_size)))
#     f_len = len(flist)
#     print(f_len)
#     # init list of files
#     batch = tf.data.Dataset.from_tensor_slices((tf.constant(flist)))
#     batch = batch.map(_pyfn_wrapper, num_parallel_calls=ncores)
#     batch = batch.batch(batch_size, drop_remainder=True).prefetch(ncores + 6).shuffle(batch_size).repeat()
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
# def _pyfn_wrapper(args):
#     return tf.py_func(parse_h5,  #wrapped pythonic function
#                       [args],
#                       [tf.float32, tf.float32]  #[input, output] dtype
#                       )
#
# def parse_h5(args):
#     name, window_size = args
#     window_size = int(window_size.decode('utf-8'))
#     with h5py.File(name, 'r') as f:
#         X = f['X'][:].reshape(window_size, window_size, 1)
#         y = f['y'][:].reshape(window_size, window_size, 1)
#         return X, y
#
#
# # init data
# # p = mp.Pool(mp.cpu_count())
# # p.map(write_h5, zip(range(9000), repeat(True)))
# # p.map(write_h5, zip(range(1000), repeat(False)))
#
# # hparam
# ep_len = 90
# step_len = 9  # run test_op after 9 steps
#
# # create tf.data.Dataset
# train_input, train_len = input_pipeline(100, 5, is_train=True)
# test_input, test_len = input_pipeline(100, 5, is_train=False)
#
#
# def model(input, reuse=True):
#     with tf.variable_scope('model', reuse=reuse):
#         with tf.name_scope("Conv1"):
#             W = tf.get_variable("W", shape=[3, 3, 1, 1],
#                                  initializer=tf.contrib.layers.xavier_initializer())
#             b = tf.get_variable("b", shape=[1], initializer=tf.contrib.layers.xavier_initializer())
#             layer1 = tf.nn.conv2d(input['img'], W, strides=[1, 1, 1, 1], padding='SAME') + b
#             logits = tf.nn.relu(layer1)
#
#         loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=input['label'], predictions=logits))
#         return loss
#
# train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(model(train_input, False))
# test_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(model(test_input, True))
#
# # session
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     for ep in range(5):
#         print('ep:{}'.format(ep))
#         sess.run(train_input['iterator_init_op'])
#         sess.run(test_input['iterator_init_op'])
#         for step in range(ep_len):
#             print('step:{}\r'.format(step))
#             try:
#                 sess.run([train_op])
#                 if step % step_len == (step_len - 1):
#                     sess.run([test_op])
#             except tf.errors.OutOfRangeError:
#                 raise('drop the remainder')