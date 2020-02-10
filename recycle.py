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

import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided
from PIL import Image
import os
import h5py
from itertools import repeat, product
from writer import _h5Writer_V2, _h5Writer_V3
from reader import _tifReader
from util import check_N_mkdir


def preprocess(indir, stride, patch_size, mode='h5', shuffle=True, evaluate=True, traintest_split_rate=0.9):
    """
    input:
    -------
        indir: (string)
        stride: (int) step of pixel
        patch_size: (int) height and width of
        mode: (string) file type of to save the preprocessed images. #TODO: .csv .tiff
        shuffle: (boolean) if True, preprocessed images will be shuffled before saving
        evaluate: (boolean) if True, preprocessed images will be saved into two directories for training set and test set
        traintest_split_rate: (float) the ratio for splitting trainning/test set
    return:
    -------
        None
    """
    # todo: can simplify to not loading in RAM
    # import data
    X_stack, y_stack, _ = _tifReader(indir)
    outdir = './proc/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    X_patches = _stride(X_stack[0], stride, patch_size)
    y_patches = _stride(y_stack[0], stride, patch_size)

    # extract patches
    for i in range(1, len(X_stack) - 1):
        X_patches = np.vstack((X_patches, _stride(X_stack[i], stride, patch_size)))
    for i in range(1, len(y_stack) - 1):
        y_patches = np.vstack((y_patches, _stride(y_stack[i], stride, patch_size)))

    assert X_patches.shape[0] == y_patches.shape[0], 'numbers of raw image: {} and label image: {} are different'.format(X_patches.shape[0], y_patches.shape[0])

    # shuffle
    if shuffle:
        X_patches, y_patches = _shuffle(X_patches, y_patches)

    if evaluate:
        if mode == 'h5':
            _h5Writer_V2(X_patches[:np.int(X_patches.shape[0] * traintest_split_rate)],
                         y_patches[:np.int(X_patches.shape[0] * traintest_split_rate)],
                         outdir + 'train/', patch_size)
            _h5Writer_V2(X_patches[np.int(X_patches.shape[0] * traintest_split_rate):],
                         y_patches[np.int(X_patches.shape[0] * traintest_split_rate):],
                         outdir + 'test/', patch_size)
        elif mode == 'csv':
            raise NotImplementedError
        else:
            raise NotImplementedError

    else:
        if mode == 'h5':
            _h5Writer_V2(X_patches, y_patches, outdir, patch_size)
        elif mode == 'csv':
            raise NotImplementedError
        else:
            raise NotImplementedError


def preprocess_V2(indir, stride, patch_size, traintest_split_rate=0.9, shuffle=False):
    """
    input:
    -------
        indir: (string)
        stride: (int) step of pixel
        patch_size: (int) height and width of
        mode: (string) file type of to save the preprocessed images. #TODO: .csv .tiff
        shuffle: (boolean) if True, preprocessed images will be shuffled before saving
        evaluate: (boolean) if True, preprocessed images will be saved into two directories for training set and test set
        traintest_split_rate: (float) the ratio for splitting trainning/test set
    return:
    -------
        None
    """
    train_outdir = './proc/train/{}/'.format(patch_size)
    test_outdir = './proc/test/{}/'.format(patch_size)
    check_N_mkdir(train_outdir)
    check_N_mkdir(test_outdir)

    # import data
    X_stack, y_stack, list_shapes = _tifReader(indir)
    assert (len(X_stack) == len(y_stack) == len(list_shapes)), 'Number of img, label, and their shapes are not equal!'

    # get ID, nb_X, nb_y
    list_ID = [id for id in range(len(list_shapes))]
    list_nb_w = [(np.asarray(X_stack[i]).shape[0] - patch_size) // stride + 1 for i in range(len(X_stack))]
    list_nb_h = [(np.asarray(y_stack[i]).shape[0] - patch_size) // stride + 1 for i in range(len(y_stack))]

    # make ID grid then pick e.g. 90% for train and 10% for test
    train_id_dict = {}
    test_id_dict = {}
    for ID, nb_w, nb_h in zip(list_ID, list_nb_w, list_nb_h):
        # build a x-y grid
        xid, yid = np.meshgrid(np.arange(nb_w), np.arange(nb_h))  # xv, yv same shape
        xid, yid = np.reshape(xid, (-1)), np.reshape(yid, (-1))  # flatten xv, yv
        tmp = np.arange(xid.size)

        # choose 90% of the pixel
        random = np.random.choice(tmp, int(xid.size * traintest_split_rate), replace=False)

        tmp = np.zeros(xid.shape)
        tmp[random] = 1
        train_id_dict[ID] = [xid[np.where(tmp == 1)], yid[np.where(tmp == 1)]]
        test_id_dict[ID] = [xid[np.where(tmp == 0)], yid[np.where(tmp == 0)]]

    # X, y coords
    # train set
    for img_id, _indir, patch_size, outdir in zip(list_ID, repeat(indir), repeat(patch_size), repeat(train_outdir)):
        _h5Writer_V3(img_ID=img_id,
                     w_ids=train_id_dict[img_id][0],
                     h_ids=train_id_dict[img_id][1],
                     in_path=indir + str(img_id),
                     stride=stride,
                     patch_size=patch_size,
                     outdir=train_outdir)

    # test set
    for img_id in list_ID:
        _h5Writer_V3(img_ID=img_id,
                     w_ids=test_id_dict[img_id][0],
                     h_ids=test_id_dict[img_id][1],
                     in_path=indir + str(img_id),
                     patch_size=patch_size,
                     stride=stride,
                     outdir=test_outdir)


def _shuffle(tensor_a, tensor_b, random_state=42):
    """
    input:
    -------
        tensor_a: (np.ndarray) input tensor
        tensor_b: (np.ndarray) input tensor
    return:
    -------
        tensor_a: (np.ndarray) shuffled tensor_a at the same way as tensor_b
        tensor_b: (np.ndarray) shuffled tensor_b at the same way as tensor_a
    """
    # shuffle two tensors in unison
    np.random.seed(random_state)
    idx = np.random.permutation(tensor_a.shape[0]) #artifacts
    return tensor_a[idx], tensor_b[idx]


def _stride(tensor, stride, patch_size):
    """
    input:
    -------
        tensor: (np.ndarray) images to stride
        stride: (int) pixel step that the window of patch jump for sampling
        patch_size: (int) height and weight (here we assume the same) of the sampling image
    return:
    -------
        patches: (np.ndarray) strided and restacked patches
    """
    p_h = (tensor.shape[0] - patch_size) // stride + 1
    p_w = (tensor.shape[1] - patch_size) // stride + 1
    # (4bytes * step * dim0, 4bytes * step, 4bytes * dim0, 4bytes)
    # stride the tensor
    _strides = tuple([i * stride for i in tensor.strides]) + tuple(tensor.strides)
    patches = as_strided(tensor, shape=(p_h, p_w, patch_size, patch_size), strides=_strides)\
        .reshape((-1, patch_size, patch_size))
    return patches


def _idParser(directory, patch_size, batch_size, mode='h5'):
    """
    input:
    -------
        directory: (string) path to be parsed
        patch_size: (int) height and weight (here we assume the same)
        batch_size: (int) number of images per batch
        mode: (string) file type to be parsed
    return:
    -------
        None
    """
    l_f = []
    max_id = 0
    # check if the .h5 with the same patch_size and batch_size exist
    for dirpath, _, fnames in os.walk(directory):
        for fname in fnames:
            if fname.split('_')[0] == patch_size and fname.split('_')[1] == batch_size and fname.endswith(mode):
                l_f.append(os.path.abspath(os.path.join(dirpath, fname)))
                max_id = max(max_id, int(fname.split('_')[2]))

    if mode == 'h5':
        try:
            with h5py.File(directory + '{}.'.format(patch_size) + mode, 'r') as f:
                rest = batch_size - f['X'].shape[0]
                return max_id, rest
        except:
            return 0, 0
    elif mode == 'csv':
        try:
            with open(directory + '{}_{}_{}.csv'.format(patch_size, batch_size, max_id) + mode, 'r') as f:
                rest = batch_size - f['X'].shape[0]
                return max_id, rest
        except:
            return 0, 0
    elif mode == 'tfrecord':
        raise NotImplementedError('tfrecord has not been implemented yet')

