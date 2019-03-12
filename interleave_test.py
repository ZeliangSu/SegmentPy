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