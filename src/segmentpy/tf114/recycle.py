# import tensorflow as tf
#
#
# # build different block
# def built_diff_block(patch_size=72):
#     """
#     inputs:
#     -------
#         patch_size: (int)
#
#     return:
#     -------
#         g_diff_def: (tf.graphdef())
#     """
#     # diff node new graph
#     with tf.Graph().as_default() as g_diff:
#         with tf.name_scope('diff_block'):
#             # two phs for passing values
#             label_ph = tf.placeholder(tf.int32, shape=[None, patch_size, patch_size, 1], name='label_ph')
#             res_ph = tf.placeholder(tf.float32, shape=[None, patch_size, patch_size, 1], name='res_ph')
#             # diff op
#             diff = tf.cast(
#                 tf.not_equal(
#                     label_ph,
#                     tf.reshape(
#                         tf.cast(
#                             res_ph,
#                             tf.int32
#                         ),
#                         [-1, patch_size, patch_size, 1]
#                     )
#                 ),
#                 tf.int32,
#                 name='diff_img'
#             )  #todo: doing like this isn't optimal
#
#         g_diff_def = g_diff.as_graph_def()
#     return g_diff_def
#
#
# def join_diff_to_mainGraph(g_diff_def, conserve_nodes, path='./dummy/pb/test.pb'):
#     """
#     inputs:
#     -------
#         g_diff_def: (tf.graphdef())
#         conserve_nodes: (list of string)
#         path: (str)
#
#     return:
#     -------
#         g_combined: (tf.Graph())
#         ops_dict: (dictionary of operations)
#     """
#     # load main graph pb
#     with tf.gfile.GFile(path, mode='rb') as f:
#         # init GraphDef()
#         restored_graph_def = tf.GraphDef()
#         # parse saved .pb to GraphDef()
#         restored_graph_def.ParseFromString(f.read())
#
#     with tf.Graph().as_default() as g_combined:
#         # import graph def
#         pred = tf.import_graph_def(
#             graph_def=restored_graph_def,
#             return_elements=[conserve_nodes[-1]],
#             name=''  #note: '' so that won't have import/ prefix
#         )
#
#         # join diff def
#         tf.import_graph_def(
#             g_diff_def,
#             input_map={'diff_block/res_ph:0': pred},
#             return_elements=['diff_block/diff_img'],
#             name='diff_block'
#         )
#
#         # prepare feed_dict for inference
#         ops_dict = {
#             'ops': [g_combined.get_tensor_by_name(op_name + ':0') for op_name in conserve_nodes],
#             'diff_img': g_combined.get_tensor_by_name('diff_block/diff_img:0'),
#         }
#
#         return g_combined, ops_dict
#
#
# def run_nodes_and_save_partial_res(g_combined, ops_dict, conserve_nodes, input_dir=None, rlt_dir=None):
#     """
#
#     Parameters
#     ----------
#     g_combined: (tf.Graph())  #note: take diff grafted graph
#     ops_dict: (list of operations)
#     conserve_nodes: (list of string)
#
#     Returns
#     -------
#         None
#
#     """
#     with g_combined.as_default() as g_combined:
#         new_input = g_combined.get_tensor_by_name('new_ph:0')
#         dropout_input = g_combined.get_tensor_by_name('input_pipeline/dropout_prob:0')
#         new_label = g_combined.get_tensor_by_name('diff_block/label_ph:0')
#         # run inference
#         with tf.Session(graph=g_combined) as sess:
#             # tf.summary.FileWriter('./dummy/tensorboard/after_combine', sess.graph)
#             print_nodes_name_shape(sess.graph)
#
#             # write firstly input and output images
#             imgs = [h5.File(input_dir + '{}.h5'.format(i))['X'] for i in range(300)]
#             _resultWriter(imgs, 'input')
#             label = [h5.File(input_dir + '{}.h5'.format(i))['y'] for i in range(300)]
#             _resultWriter(label, 'label')
#             img_size = np.array(Image.open(imgs[0])).shape[1]
#
#             feed_dict = {
#                 new_input: np.array(imgs).reshape((300, img_size, img_size, 1)),
#                 dropout_input: 1.0,
#                 new_label: np.array(label).reshape((300, img_size, img_size, 1)),
#             }
#
#             # run partial results operations and diff block
#             res, res_diff = sess.run([ops_dict['ops'], ops_dict['diff_img']], feed_dict=feed_dict)
#
#             # note: save partial/final inferences of the first image
#             for layer_name, tensors in zip(conserve_nodes, res):
#                 if tensors.ndim == 4 or 2:
#                     tensors = tensors[0]  # todo: should generalize to batch
#                 _resultWriter(tensors, layer_name=layer_name.split('/')[-2],
#                               path=rlt_dir)  # for cnn outputs shape: [batch, w, h, nb_conv]
#
#             # note: save diff of all imgs
#             _resultWriter(np.transpose(np.squeeze(res_diff), (1, 2, 0)), 'diff',
#                           path=rlt_dir)  # for diff output shape: [batch, w, h, 1]


# # further convert to tflite
# check_N_mkdir(paths['tflite_pb_dir'])
# input_arrays = ['input_ph', 'dropout_ph']
# input_shapes = {'input_ph': [10, 72, 72, 1], 'dropout_ph': [1]}
# output_arrays = [conserve_nodes[0]]
# converter = tf.lite.TFLiteConverter.from_frozen_graph(
#         paths['optimized_pb_path'], input_arrays=input_arrays,
#     output_arrays=output_arrays, input_shapes=input_shapes
#     )
# converter.allow_custom_ops = True  #note: this line avoid throwing up error RandomUniform, NearestNeighbour, Leaky
# tflite_model = converter.convert()
# open(paths['tflite_pb_dir'] + "converted_model.tflite", "wb").write(tflite_model)

# # perform inference
# # Load TFLite model and allocate tensors.
# interpreter = tf.lite.Interpreter(model_path="./model/tf113/tflite/converted_model.tflite")  #note: ValueError: Didn't find custom op for name 'RandomUniform' with version 1 Registration failed.
# interpreter.allocate_tensors()
#
# # Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()
#
# # Test model on random input data.
# input_shape = input_details[0]['shape']
# interpreter.set_tensor(input_details[0]['index'], np.zeros((10, 72, 72, 1)))
#
# interpreter.invoke()
#
# # The function `get_tensor()` returns a copy of the tensor data.
# # Use `tensor()` in order to get a pointer to the tensor.
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)

# import tensorflow as tf
#
#
# def pb_to_pbtxt(path):
#     with tf.gfile.FastGFile(path, 'rb') as f:
#         graph_def = tf.GraphDef()
#         graph_def.ParseFromString(f.read())
#         tf.import_graph_def(graph_def, name='')
#         tf.train.write_graph(graph_def, './pbtxt/', 'savedmodel.pbtxt', as_text=True)
#         print(graph_def)
#     return
#
#
# if __name__ == '__main__':
#     graph_def_dir = './logs/2019_10_8_bs300_ps72_lr0.0001_cs5_nc80_act_leaky_aug_True/hour9_1st_try_end1epBUG/'
#     graph_def_file = graph_def_dir + 'savedmodel/step2000/saved_model.pb'
#     ckpt_converted_model = graph_def_dir + 'ckpt_converted_model.pb'
#     in_tensor = 'input_pipeline/input_cond/Merge_1'
#     out_tensor = 'model/decoder/logits/relu'
#
#     # convert ckpt to pb
#     restorer = tf.train.import_meta_graph(
#         graph_def_dir + 'ckpt/step2000.meta',
#     )
#     input_graph_def = tf.get_default_graph().as_graph_def()
#     with tf.Session() as sess: \
#         # note: TF1.13 only step2000 not step2000.ckpt
#         restorer.restore(sess,
#                          graph_def_dir + 'ckpt/step2000'
#                          )
#         output_graph_def = tf.graph_util.convert_variables_to_constants(
#             sess=sess,  # note: variables are always in a session
#             input_graph_def=input_graph_def,
#             output_node_names=[in_tensor, out_tensor],
#         )
#     with tf.gfile.GFile(ckpt_converted_model, 'wb') as f:  # 'wb' stands for write binary
#         f.write(output_graph_def.SerializeToString())
#
#
#     # pb_to_pbtxt(graph_def_file)  #fixme: Ignore 'tcmalloc: large alloc' warnings.
#     # Traceback (most recent call last):
#     #   File "/home/tomoserver/anaconda3/envs/tf113/lib/python3.6/site-packages/tensorflow/lite/python/lite.py", line 254, in from_frozen_graph
#     #     graph_def.ParseFromString(file_content)
#     # google.protobuf.message.DecodeError: Error parsing message
#     #
#     # During handling of the above exception, another exception occurred:
#
#     # convert Graph Def to tflite file
#     converter = tf.lite.TFLiteConverter.from_frozen_graph(
#         ckpt_converted_model, [in_tensor], [out_tensor],
#         input_shapes={in_tensor: []}
#     )
#     tflite_model = converter.convert()
#     with open(graph_def_dir + 'saved_model.tflite', 'wb') as f:
#         f.write(tflite_model)
#
#
#     # perform inference


################################
#
#        proc.py
#
################################
#
# import tensorflow as tf
# import numpy as np
# from numpy.lib.stride_tricks import as_strided
# from PIL import Image
# import os
# import h5py
# from itertools import repeat, product
# from writer import _h5Writer_V2, _h5Writer_V3
# from reader import _tifReader
# from util import check_N_mkdir
#
#
# def preprocess(indir, stride, patch_size, mode='h5', shuffle=True, evaluate=True, traintest_split_rate=0.9):
#     """
#     input:
#     -------
#         indir: (string)
#         stride: (int) step of pixel
#         patch_size: (int) height and width of
#         mode: (string) file type of to save the preprocessed images. #TODO: .csv .tiff
#         shuffle: (boolean) if True, preprocessed images will be shuffled before saving
#         evaluate: (boolean) if True, preprocessed images will be saved into two directories for training set and test set
#         traintest_split_rate: (float) the ratio for splitting trainning/test set
#     return:
#     -------
#         None
#     """
#     # todo: can simplify to not loading in RAM
#     # import data
#     X_stack, y_stack, _ = _tifReader(indir)
#     outdir = './proc/'
#     if not os.path.exists(outdir):
#         os.mkdir(outdir)
#
#     X_patches = _stride(X_stack[0], stride, patch_size)
#     y_patches = _stride(y_stack[0], stride, patch_size)
#
#     # extract patches
#     for i in range(1, len(X_stack) - 1):
#         X_patches = np.vstack((X_patches, _stride(X_stack[i], stride, patch_size)))
#     for i in range(1, len(y_stack) - 1):
#         y_patches = np.vstack((y_patches, _stride(y_stack[i], stride, patch_size)))
#
#     assert X_patches.shape[0] == y_patches.shape[0], 'numbers of raw image: {} and label image: {} are different'.format(X_patches.shape[0], y_patches.shape[0])
#
#     # shuffle
#     if shuffle:
#         X_patches, y_patches = _shuffle(X_patches, y_patches)
#
#     if evaluate:
#         if mode == 'h5':
#             _h5Writer_V2(X_patches[:np.int(X_patches.shape[0] * traintest_split_rate)],
#                          y_patches[:np.int(X_patches.shape[0] * traintest_split_rate)],
#                          outdir + 'train/', patch_size)
#             _h5Writer_V2(X_patches[np.int(X_patches.shape[0] * traintest_split_rate):],
#                          y_patches[np.int(X_patches.shape[0] * traintest_split_rate):],
#                          outdir + 'test/', patch_size)
#         elif mode == 'csv':
#             raise NotImplementedError
#         else:
#             raise NotImplementedError
#
#     else:
#         if mode == 'h5':
#             _h5Writer_V2(X_patches, y_patches, outdir, patch_size)
#         elif mode == 'csv':
#             raise NotImplementedError
#         else:
#             raise NotImplementedError
#
#
# def preprocess_V2(indir, stride, patch_size, traintest_split_rate=0.9, shuffle=False):
#     """
#     input:
#     -------
#         indir: (string)
#         stride: (int) step of pixel
#         patch_size: (int) height and width of
#         mode: (string) file type of to save the preprocessed images. #TODO: .csv .tiff
#         shuffle: (boolean) if True, preprocessed images will be shuffled before saving
#         evaluate: (boolean) if True, preprocessed images will be saved into two directories for training set and test set
#         traintest_split_rate: (float) the ratio for splitting trainning/test set
#     return:
#     -------
#         None
#     """
#     train_outdir = './proc/train/{}/'.format(patch_size)
#     test_outdir = './proc/test/{}/'.format(patch_size)
#     check_N_mkdir(train_outdir)
#     check_N_mkdir(test_outdir)
#
#     # import data
#     X_stack, y_stack, list_shapes = _tifReader(indir)
#     assert (len(X_stack) == len(y_stack) == len(list_shapes)), 'Number of img, label, and their shapes are not equal!'
#
#     # get ID, nb_X, nb_y
#     list_ID = [id for id in range(len(list_shapes))]
#     list_nb_w = [(np.asarray(X_stack[i]).shape[0] - patch_size) // stride + 1 for i in range(len(X_stack))]
#     list_nb_h = [(np.asarray(y_stack[i]).shape[0] - patch_size) // stride + 1 for i in range(len(y_stack))]
#
#     # make ID grid then pick e.g. 90% for train and 10% for test
#     train_id_dict = {}
#     test_id_dict = {}
#     for ID, nb_w, nb_h in zip(list_ID, list_nb_w, list_nb_h):
#         # build a x-y grid
#         xid, yid = np.meshgrid(np.arange(nb_w), np.arange(nb_h))  # xv, yv same shape
#         xid, yid = np.reshape(xid, (-1)), np.reshape(yid, (-1))  # flatten xv, yv
#         tmp = np.arange(xid.size)
#
#         # choose 90% of the pixel
#         random = np.random.choice(tmp, int(xid.size * traintest_split_rate), replace=False)
#
#         tmp = np.zeros(xid.shape)
#         tmp[random] = 1
#         train_id_dict[ID] = [xid[np.where(tmp == 1)], yid[np.where(tmp == 1)]]
#         test_id_dict[ID] = [xid[np.where(tmp == 0)], yid[np.where(tmp == 0)]]
#
#     # X, y coords
#     # train set
#     for img_id, _indir, patch_size, outdir in zip(list_ID, repeat(indir), repeat(patch_size), repeat(train_outdir)):
#         _h5Writer_V3(img_ID=img_id,
#                      w_ids=train_id_dict[img_id][0],
#                      h_ids=train_id_dict[img_id][1],
#                      in_path=indir + str(img_id),
#                      stride=stride,
#                      patch_size=patch_size,
#                      outdir=train_outdir)
#
#     # test set
#     for img_id in list_ID:
#         _h5Writer_V3(img_ID=img_id,
#                      w_ids=test_id_dict[img_id][0],
#                      h_ids=test_id_dict[img_id][1],
#                      in_path=indir + str(img_id),
#                      patch_size=patch_size,
#                      stride=stride,
#                      outdir=test_outdir)
#
#
# def _shuffle(tensor_a, tensor_b, random_state=42):
#     """
#     input:
#     -------
#         tensor_a: (np.ndarray) input tensor
#         tensor_b: (np.ndarray) input tensor
#     return:
#     -------
#         tensor_a: (np.ndarray) shuffled tensor_a at the same way as tensor_b
#         tensor_b: (np.ndarray) shuffled tensor_b at the same way as tensor_a
#     """
#     # shuffle two tensors in unison
#     np.random.seed(random_state)
#     idx = np.random.permutation(tensor_a.shape[0]) #artifacts
#     return tensor_a[idx], tensor_b[idx]
#
#
# def _stride(tensor, stride, patch_size):
#     """
#     input:
#     -------
#         tensor: (np.ndarray) images to stride
#         stride: (int) pixel step that the window of patch jump for sampling
#         patch_size: (int) height and weight (here we assume the same) of the sampling image
#     return:
#     -------
#         patches: (np.ndarray) strided and restacked patches
#     """
#     p_h = (tensor.shape[0] - patch_size) // stride + 1
#     p_w = (tensor.shape[1] - patch_size) // stride + 1
#     # (4bytes * step * dim0, 4bytes * step, 4bytes * dim0, 4bytes)
#     # stride the tensor
#     _strides = tuple([i * stride for i in tensor.strides]) + tuple(tensor.strides)
#     patches = as_strided(tensor, shape=(p_h, p_w, patch_size, patch_size), strides=_strides)\
#         .reshape((-1, patch_size, patch_size))
#     return patches
#
#
# def _idParser(directory, patch_size, batch_size, mode='h5'):
#     """
#     input:
#     -------
#         directory: (string) path to be parsed
#         patch_size: (int) height and weight (here we assume the same)
#         batch_size: (int) number of images per batch
#         mode: (string) file type to be parsed
#     return:
#     -------
#         None
#     """
#     l_f = []
#     max_id = 0
#     # check if the .h5 with the same patch_size and batch_size exist
#     for dirpath, _, fnames in os.walk(directory):
#         for fname in fnames:
#             if fname.split('_')[0] == patch_size and fname.split('_')[1] == batch_size and fname.endswith(mode):
#                 l_f.append(os.path.abspath(os.path.join(dirpath, fname)))
#                 max_id = max(max_id, int(fname.split('_')[2]))
#
#     if mode == 'h5':
#         try:
#             with h5py.File(directory + '{}.'.format(patch_size) + mode, 'r') as f:
#                 rest = batch_size - f['X'].shape[0]
#                 return max_id, rest
#         except:
#             return 0, 0
#     elif mode == 'csv':
#         try:
#             with open(directory + '{}_{}_{}.csv'.format(patch_size, batch_size, max_id) + mode, 'r') as f:
#                 rest = batch_size - f['X'].shape[0]
#                 return max_id, rest
#         except:
#             return 0, 0
#     elif mode == 'tfrecord':
#         raise NotImplementedError('tfrecord has not been implemented yet')
#
#
# ###########################
# #
# #      inference.py
# #
# ###########################
#
#
# def inference_recursive(inputs=None, conserve_nodes=None, paths=None, hyper=None):
#     assert isinstance(conserve_nodes, list), 'conserve nodes should be a list'
#     assert isinstance(inputs, list), 'inputs is expected to be a list of images for heterogeneous image size!'
#     assert isinstance(paths, dict), 'paths should be a dict'
#     assert isinstance(hyper, dict), 'hyper should be a dict'
#     check_N_mkdir(paths['out_dir'])
#     freeze_ckpt_for_inference(paths=paths, hyper=hyper, conserve_nodes=conserve_nodes)  # there's still some residual nodes
#     optimize_pb_for_inference(paths=paths, conserve_nodes=conserve_nodes)  # clean residual nodes: gradients, td.data.pipeline...
#
#     # set device
#     config_params = {}
#     if hyper['device_option'] == 'cpu':
#         config_params['config'] = tf.ConfigProto(device_count={'GPU': 0})
#     elif 'specific' in hyper['device_option']:
#         print('using GPU:{}'.format(hyper['device_option'].split(':')[-1]))
#         config_params['config'] = tf.ConfigProto(
#             gpu_options=tf.GPUOptions(visible_device_list=hyper['device_option'].split(':')[-1]),
#             allow_soft_placement=True,
#             log_device_placement=False,
#             )
#
#     # load graph
#     tf.reset_default_graph()
#     with tf.gfile.GFile(paths['optimized_pb_path'], 'rb') as f:
#         graph_def_optimized = tf.GraphDef()
#         graph_def_optimized.ParseFromString(f.read())
#     l_out = []
#
#     with tf.Session(**config_params) as sess:
#         #note: ValueError: Input 0 of node import/model/contractor/conv1/conv1_2/batch_norm/cond/Switch was passed float from import/new_BN_phase:0 incompatible with expected bool.
#         # WARNING:tensorflow:Didn't find expected Conv2D input to 'model/contractor/conv1/conv1_2/batch_norm/cond/FusedBatchNorm'
#         # WARNING:tensorflow:Didn't find expected Conv2D input to 'model/contractor/conv1/conv1_2/batch_norm/cond/FusedBatchNorm_1'
#         # print(graph_def_optimized.node)
#         _ = tf.import_graph_def(graph_def_optimized, return_elements=[conserve_nodes[-1]])
#         G = tf.get_default_graph()
#         tf.summary.FileWriter(paths['working_dir'] + 'tb/after_optimized', sess.graph)
#         # print_nodes_name(G)
#         #todo: replacee X with a inputpipeline
#         X = G.get_tensor_by_name('import/' + 'new_input:0')
#         y = G.get_tensor_by_name('import/' + conserve_nodes[-1] + ':0')
#         bn = G.get_tensor_by_name('import/' + 'new_BN:0')  #note: not needed anymore
#         do = G.get_tensor_by_name('import/' + 'new_dropout:0')  #note: not needed anymore
#
#         # compute the dimensions of the patches array
#         for i, _input in tqdm(enumerate(inputs), desc='image'):
#
#             # use reconstructor to not saturate the RAM
#             if hyper['mode'] == 'classification':
#                 output = reconstructor_V2_cls(_input.shape, hyper['patch_size'], hyper['stride'], y.shape[3])
#             else:
#                 output = reconstructor_V2_reg(_input.shape, hyper['patch_size'], hyper['stride'])
#
#             n_h, n_w = output.get_nb_patch()
#             hyper['nb_batch'] = n_h * n_w // hyper['batch_size']
#             last_batch_len = n_h * n_w % hyper['batch_size']
#             logger.info('\nnumber of batch: {}'.format(hyper['nb_batch']))
#             logger.info('\nbatch size: {}'.format(hyper['batch_size']))
#             logger.info('\nlast batch size: {}'.format(last_batch_len))
#
#             # inference
#             for i_batch in tqdm(range(hyper['nb_batch'] + 1), desc='batch'):
#                 if i_batch < hyper['nb_batch']:
#                     start_id = i_batch * hyper['batch_size']
#                     batch = []
#                     # construct input
#                     id_list = np.arange(start_id, start_id + hyper['batch_size'], 1)
#                     for id in np.nditer(id_list):
#                         b = id // n_h
#                         a = id % n_h
#                         logger.debug('\n id: {}'.format(id))
#                         logger.debug('\n row coordinations: {} - {}'.format(a * hyper['stride'], a * hyper['stride'] + hyper['patch_size']))
#                         logger.debug('\n colomn coordinations: {} - {}'.format(b * hyper['stride'], b * hyper['stride'] + hyper['patch_size']))
#                         # concat patch to batch
#                         batch.append(_input[
#                                      a * hyper['stride']: a * hyper['stride'] + hyper['patch_size'],
#                                      b * hyper['stride']: b * hyper['stride'] + hyper['patch_size']
#                                      ])
#
#                     # inference
#                     # batch = np.asarray(_minmaxscalar(batch))  #note: don't forget the minmaxscalar, since during training we put it
#                     batch = np.expand_dims(batch, axis=3)  # ==> (8, 512, 512, 1)
#                     feed_dict = {
#                         X: batch,
#                         do: 1.0,
#                         bn: False,
#                     }
#                     _out = sess.run(y, feed_dict=feed_dict)
#                     if hyper['mode'] == 'classification':
#                         _out = customized_softmax_np(_out)
#                     output.add_batch(_out, start_id)
#                 else:
#                     if last_batch_len != 0:
#                         logger.info('last batch')
#                         start_id = i_batch * hyper['batch_size']
#                         batch = []
#                         # construct input
#                         id_list = np.arange(start_id, start_id + last_batch_len, 1)
#                         for id in np.nditer(id_list):
#                             b = id // n_h
#                             a = id % n_h
#                             batch.append(_input[
#                                          a * hyper['stride']: a * hyper['stride'] + hyper['patch_size'],
#                                          b * hyper['stride']: b * hyper['stride'] + hyper['patch_size']
#                                          ])
#
#                         # 0-padding batch
#                         batch = np.asarray(_minmaxscalar(batch))
#                         batch = np.expand_dims(batch, axis=3)
#                         batch = np.concatenate([batch, np.ones(
#                             (hyper['batch_size'] - last_batch_len, *batch.shape[1:])
#                         )], axis=0)
#
#                         feed_dict = {
#                             X: batch,
#                             do: 1.0,  #note: not needed anymore
#                             bn: False,  #note: not needed anymore
#                         }
#                         _out = sess.run(y, feed_dict=feed_dict)
#                         if hyper['mode'] == 'classification':
#                             _out = customized_softmax_np(_out)
#                         output.add_batch(_out[:last_batch_len], start_id)
#
#             # reconstruction
#             output.reconstruct()
#             output = output.get_reconstruction()
#             # save
#             check_N_mkdir(paths['out_dir'])
#             output = np.squeeze(output)
#             Image.fromarray(output.astype(np.float32)).save(paths['out_dir'] + 'step{}_{}.tif'.format(paths['step'], i))
#             l_out.append(output)
#     return l_out
#
#
# def reconstruct(stack, image_size=None, stride=None):
#     """
#     inputs:
#     -------
#         stack: (np.ndarray) stack of patches to reconstruct
#         image_size: (tuple | list) height and width for the final reconstructed image
#         stride: (int) herein should be the SAME stride step that one used for preprocess
#     return:
#     -------
#         img: (np.ndarray) final reconstructed image
#         nb_patches: (int) number of patches need to provide to this function
#     """
#     i_h, i_w = image_size[:2]  #e.g. (a, b)
#     p_h, p_w = stack.shape[1:3]  #e.g. (x, h, w, 1)
#     img = np.zeros((i_h, i_w))
#
#     # compute the dimensions of the patches array
#     n_h = (i_h - p_h) // stride + 1
#     n_w = (i_w - p_w) // stride + 1
#
#     for p, (i, j) in zip(stack, product(range(n_h), range(n_w))):
#         img[i * stride:i * stride + p_h, j * stride:j * stride + p_w] += p
#
#     for i in range(i_h):
#         for j in range(i_w):
#             img[i, j] /= float(min(i + stride, p_h, i_h - i) *
#                                min(j + stride, p_w, i_w - j))
#     return img
#
# def inputpipeline(batch_size, ncores=mp.cpu_count(), suffix='', augmentation=False, mode='regression'):
#     """
#     tensorflow tf.data input pipeline based helper that return image and label at once
#
#     input:
#     -------
#         batch_size: (int) number of images per batch before update parameters
#
#     output:
#     -------
#         inputs: (dict) output of this func, but inputs of the neural network. A dictionary of img, label and the iterator
#         initialization operation
#     """
#
#     warnings.warn('The tf.py_func() will be deprecated at TF2.0, replaced by tf.function() please change later the inputpipeline() in input.py')
#
#     is_training = True if suffix in ['train', 'cv', 'test'] else False
#
#     if is_training:
#         # placeholder for list fo files
#         with tf.name_scope('input_pipeline_' + suffix):
#             fnames_ph = tf.placeholder(tf.string, shape=[None], name='fnames_ph')
#             patch_size_ph = tf.placeholder(tf.int32, shape=[None], name='patch_size_ph')
#
#             # init and shuffle list of files
#             batch = tf.data.Dataset.from_tensor_slices((fnames_ph, patch_size_ph))
#             batch = batch.shuffle(tf.cast(tf.shape(fnames_ph)[0], tf.int64))
#             # read data
#             if mode == 'regression':
#                 batch = batch.map(_pyfn_regression_parser_wrapper, num_parallel_calls=ncores)
#             elif mode == 'classification':
#                 batch = batch.map(_pyfn_classification_parser_wrapper, num_parallel_calls=ncores)
#             # random augment data
#             if augmentation:
#                 batch = batch.map(_pyfn_aug_wrapper, num_parallel_calls=ncores)
#             # shuffle and prefetch batch
#             batch = batch.shuffle(batch_size).batch(batch_size, drop_remainder=True).prefetch(ncores).repeat()
#
#             # todo: prefetch_to_device
#             # batch = batch.apply(tf.data.experimental.prefetch_to_device('/device:GPU:0'))
#
#             # construct iterator
#             it = tf.data.Iterator.from_structure(batch.output_types, batch.output_shapes)
#             iter_init_op = it.make_initializer(batch, name='iter_init_op')
#             # get next img and label
#             X_it, y_it = it.get_next()
#
#             # dict
#             inputs = {'img': X_it,
#                       'label': y_it,
#                       'iterator_init_op': iter_init_op,
#                       'fnames_ph': fnames_ph,
#                       'patch_size_ph': patch_size_ph}
#
#     else:
#         raise NotImplementedError('Inference input need to be debug')
#     return inputs
#
#
# def parse_h5_one_hot(fname, patch_size):
#     with h5py.File(fname.decode('utf-8', 'r'), 'r') as f:
#         X = f['X'][:].reshape(patch_size, patch_size, 1)
#         y = f['y'][:].reshape(patch_size, patch_size, 1)
#
#         # if y is saved as float, convert to int
#
#         # note: {0, 50} might better separate two peaks? but not too difficult to converge at the beginning
#         y = _one_hot(y)
#         logger.debug('y shape: {}, nb_class: {}'.format(y.shape, y.shape[-1]))  #B, H, W, C
#
#         # return _minmaxscalar(X), y.astype(np.int32)  #note: minmaxscal will alternate if not all classes are present
#         return X, y.astype(np.int32)
#
#
# def _pyfn_classification_parser_wrapper(fname, patch_size):
#     return tf.py_func(
#         parse_h5_one_hot,
#         [fname, patch_size],
#         [tf.float32, tf.int32]
#     )
#
#
# import numpy as np
# import pandas as pd
# from scipy.interpolate import interp2d
# import matplotlib.pyplot as plt
# from util import clean
#
# if_log = False
# if_plot = True
# if_clean_data = False
# l_steps = ['0000', 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]
# l_input_path = ['./dummy/lss_step{}.csv'.format(step) for step in l_steps]
# l_output_path = ['./dummy/lss_step{}_interp{}.csv'.format(step, '_log' if if_log else None) for step in l_steps]
#
# ##################
# #
# #  conversion
# #
# ##################
#
#
# def csv_interp(x_mesh, y_mesh, metrics_tensor, out_path, interp_scope=5):
#     new_xmesh = np.linspace(np.min(x_mesh), np.max(x_mesh), interp_scope * x_mesh.shape[0])
#     new_ymesh = np.linspace(np.min(y_mesh), np.max(y_mesh), interp_scope * x_mesh.shape[1])
#     newxx, newyy = np.meshgrid(new_xmesh, new_ymesh)
#
#     # interpolation
#     interpolation = interp2d(x_mesh, y_mesh, metrics_tensor, kind='linear')  #note: cubic doesn't work
#     zval = interpolation(new_xmesh, new_ymesh)
#     pd.DataFrame({'xcoord': newxx.ravel(),
#                   'ycoord': newyy.ravel(),
#                   'zval': zval.ravel()}
#                  ).to_csv(out_path, index=False, header=True)
#
#
# for input_path, output_path in zip(l_input_path, l_output_path):
#     lss = np.asarray(pd.read_csv(input_path))
#     print('Loss/Acc range: {} - {}'.format(np.min(lss), np.max(lss)))
#     print('Out range: {} - {}'.format(np.min(np.log(lss)), np.max(np.log(lss))))
#     if if_log:
#         lss = np.log(lss)
#     if if_clean_data:
#         lss = clean(lss)
#     x_mesh = np.linspace(-1, 1, 51)
#     y_mesh = np.linspace(-1, 1, 51)
#     xx, yy = np.meshgrid(x_mesh, y_mesh)
#
#     csv_interp(xx, yy, lss, output_path)
#
#     ##################
#     #
#     # plot
#     #
#     ##################
#     if if_plot:
#         lss_interp = np.asarray(pd.read_csv(output_path))
#         x_mesh = np.linspace(-1, 1, 51 * 5)
#         y_mesh = np.linspace(-1, 1, 51 * 5)
#         xx, yy = np.meshgrid(x_mesh, y_mesh)
#
#         fig, ax = plt.subplots(1)
#         cs = ax.contour(xx, yy, lss_interp[:, -1].reshape(255, 255))
#         plt.clabel(cs, inline=1, fontsize=10)
#         plt.savefig(output_path.replace('.csv', '.png'))
#         plt.show()
#
# ###############
# #
# # h5.py
# #
# ###############
# import matplotlib.pyplot as plt
# import h5py
#
# with h5py.File('./proc/40.h5', 'r') as f:
#     print(f['X'].shape)
#     for i in range(10):
#         plt.figure(i)
#         plt.imshow(f['X'][i*500, ])
#         plt.figure(i + 6)
#         plt.imshow(f['y'][i*500, ])
#         plt.show()
#
#
# #############
# #
# # writer.py
# #
# #############
# def _h5Writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId, mode='onefile'):
#     patch_shape = (patch_size, patch_size)
#     # fill last .h5
#     if mode == 'h5s':
#         _h5s_writer(X_patches, y_patches, patch_shape, id_length, rest, outdir, patch_size, batch_size, maxId)
#     elif mode == 'h5':
#         _h5_writer(X_patches, y_patches, patch_shape, outdir, patch_size)
#     elif mode == 'csvs':
#         _csvs_writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId)
#     elif mode == 'tfrecord':
#         raise NotImplementedError("tfrecords part hasn't been implemented yet")
#     else:
#         raise ValueError("Please choose a mode from h5, csv or tfrecord!")
#
#
# def _h5Writer_V2(X_patches, y_patches, outdir, patch_size):
#     import os
#     if not os.path.exists(outdir):
#         os.mkdir(outdir)
#
#     if not os.path.exists('{}{}'.format(outdir, patch_size)):
#         os.mkdir('{}{}'.format(outdir, patch_size))
#
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         pool.starmap(_writer_V2, ((X_patches[i], y_patches[i], outdir, i, patch_size) for i in range(X_patches.shape[0])))
#
#
# def _writer_V2(X, y, outdir, name, patch_size):
#     with h5py.File('{}{}/{}.h5'.format(outdir, patch_size, name), 'w') as f:
#         f.create_dataset('X', (patch_size, patch_size), dtype='float32', data=X)
#         f.create_dataset('y', (patch_size, patch_size), dtype='float32', data=y)
#
#
# def _h5Writer_V3(img_ID, w_ids, h_ids, in_path, patch_size, stride, outdir):
#     assert isinstance(img_ID, int), 'Param ID should be interger'
#     assert isinstance(stride, int), 'Stride should be interger'
#     assert isinstance(w_ids, np.ndarray), 'Param ID should be np array'
#     assert isinstance(h_ids, np.ndarray), 'Param ID should be np array'
#     assert isinstance(in_path, str), 'Param ID should be np array'
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         pool.starmap(_writer_V3, ((ID, xid, yid, _in_path, _outdir, stride, _patch_size)
#                                   for ID, xid, yid, _in_path, _outdir, _patch_size, stride
#                                   in zip(repeat(img_ID), w_ids, h_ids, repeat(in_path), repeat(outdir), repeat(patch_size), repeat(stride))))
#
#
# def _writer_V3(img_ID, x_id, y_id, in_path, outdir, stride, patch_size):
#     logger.debug(mp.current_process())
#     X = np.asarray(Image.open(in_path + '.tif', 'r'))[x_id * stride: x_id * stride + patch_size, y_id * stride: y_id * stride + patch_size]
#     y = np.asarray(Image.open(in_path + '_label.tif', 'r'))[x_id * stride: x_id * stride + patch_size, y_id * stride: y_id * stride + patch_size]
#     with h5py.File('{}/{}_{}_{}.h5'.format(outdir, img_ID, x_id, y_id), 'w') as f:
#         f.create_dataset('X', (patch_size, patch_size), dtype='float32', data=X)
#         f.create_dataset('y', (patch_size, patch_size), dtype='float32', data=y)
#
#
# def _csvs_writer(X_patches, y_patches, id_length, rest, outdir, patch_size, batch_size, maxId,):
#     print('This will generate {} .csv in /proc/ directory'.format(id_length))
#     if rest > 0:
#         with open(outdir + 'X{}_{}_{}.csv'.format(patch_size, batch_size, maxId), 'ab') as f:
#             np.savetxt(f, X_patches[:rest].ravel(), delimiter=',')  #csv can only save 1d or 2d array, we reshape on reading
#         with open(outdir + 'y{}_{}_{}.csv'.format(patch_size, batch_size, maxId), 'a') as f:
#             np.savetxt(f, y_patches[:rest].ravel(), delimiter=',')
#
#     else:
#         # then create new .h5
#         for id in np.nditer(np.linspace(maxId, maxId + id_length, id_length, dtype='int')):
#             try:
#                 start = rest + batch_size * (id - maxId)
#                 end = rest + batch_size * (id - maxId + 1)
#                 with open(outdir + 'X{}_{}_{}.csv'.format(patch_size, batch_size, id), 'wb') as f:
#                     np.savetxt(f, X_patches[start:end, ].ravel(), delimiter=',')
#                 with open(outdir + 'y{}_{}_{}.csv'.format(patch_size, batch_size, id), 'wb') as f:
#                     np.savetxt(f, y_patches[start:end, ].ravel(), delimiter=',')
#             except Exception as e:
#                 print(e)
#                 # if the last one can't complete the whole .h5 file
#                 mod = (X_patches.shape[0] - rest) % batch_size
#                 with open(outdir + 'X{}_{}_{}.csv'.format(patch_size, batch_size, id), 'wb') as f:
#                     np.savetxt(f, X_patches[-mod:].ravel(), delimiter=',')
#                 with open(outdir + 'y{}_{}_{}.csv'.format(patch_size, batch_size, id), 'wb') as f:
#                     np.savetxt(f, y_patches[-mod:].ravel(), delimiter=',')
#
#
# def _h5s_writer(X_patches, y_patches, patch_shape, id_length, rest, outdir, patch_size, batch_size, maxId,):
#     print('This will generate {} .h5 in /proc/ directory'.format(id_length))
#     if rest > 0:
#         if len(X_patches.shape[0]) < rest:
#             with h5py.File(outdir + '{}_{}_{}.h5'.format(patch_size, batch_size, maxId), 'a') as f:
#                 f['X'].resize(f['X'].shape[0] + rest, axis=0)
#                 f['y'].resize(f['y'].shape[0] + rest, axis=0)
#                 f['X'][-rest:] = X_patches[:rest]
#                 f['y'][-rest:] = y_patches[:rest]
#         else:
#             with h5py.File(outdir + '{}_{}_{}.h5'.format(patch_size, batch_size, maxId), 'a') as f:
#                 f['X'][-rest:] = X_patches[:rest]
#                 f['y'][-rest:] = y_patches[:rest]
#     else:
#         # then create new .h5
#         for id in np.nditer(np.linspace(maxId, maxId + id_length, id_length, dtype='int')):
#             try:
#                 with h5py.File(outdir + '{}_{}_{}.h5'.format(patch_size, batch_size, id), 'w') as f:
#                     start = rest + batch_size * (id - maxId)
#                     end = rest + batch_size * (id - maxId) + batch_size
#
#                     X = f.create_dataset('X', (batch_size, *patch_shape),
#                                          maxshape=(None, *patch_shape),
#                                          dtype='float32')
#                     X[:] = X_patches[start:end, ]
#                     y = f.create_dataset('y', (batch_size, *patch_shape),
#                                          maxshape=(None, *patch_shape),
#                                          dtype='int8')
#                     y[:] = y_patches[start:end, ]
#             except:
#                 # if the last one can't complete the whole .h5 file
#                 with h5py.File(outdir + '{}_{}_{}.h5'.format(patch_size, batch_size, id), 'w') as f:
#                     mod = (X_patches.shape[0] - rest) % batch_size
#                     X = f.create_dataset('X', (batch_size, *patch_shape),
#                                          maxshape=(None, *patch_shape),
#                                          dtype='float32')
#                     X[mod:] = X_patches[-mod:]
#                     y = f.create_dataset('y', (batch_size, *patch_shape),
#                                          maxshape=(None, *patch_shape),
#                                          dtype='int8')
#                     y[mod:] = y_patches[-mod:]
#
#
# def _h5_writer(X_patches, y_patches, patch_shape, outdir, patch_size):
#     try:
#         with h5py.File(outdir + '{}.h5'.format(patch_size), 'a') as f:
#             f['X'].resize(f['X'].shape[0] + X_patches.shape[0], axis=0)
#             f['y'].resize(f['y'].shape[0] + y_patches.shape[0], axis=0)
#             f['X'][-X_patches.shape[0]:] = X_patches[:X_patches.shape[0]]
#             f['y'][-y_patches.shape[0]:] = y_patches[:y_patches.shape[0]]
#         print('\n***Appended patches in .h5')
#
#     except:
#         with h5py.File(outdir + '{}.h5'.format(patch_size), 'w') as f:
#             X = f.create_dataset('X', (X_patches.shape[0], *patch_shape),
#                                  maxshape=(None, *patch_shape),
#                                  dtype='float32')
#             X[:] = X_patches[:X_patches.shape[0], ]
#             y = f.create_dataset('y', (y_patches.shape[0], *patch_shape),
#                                  maxshape=(None, *patch_shape),
#                                  dtype='int8')
#             y[:] = y_patches[:y_patches.shape[0], ]
#         print('\n***Created new .h5')
#
