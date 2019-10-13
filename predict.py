import tensorflow as tf
from visualize import convert_ckpt2pb, load_mainGraph
from proc import _stride
from input import inputpipeline
from util import print_nodes_name
import numpy as np
from itertools import product
from PIL import Image
from tqdm import tqdm
import os
import timeit


def greff_pipeline_to_mainGraph(pipeline, path='./dummy/pb/test.pb'):
    """
    inputs:
    -------
        g_diff_def: (tf.graphdef())
        conserve_nodes: (list of string)
        path: (str)

    return:
    -------
        g_combined: (tf.Graph())
        ops_dict: (dictionary of operations)
    """
    # load main graph pb
    with tf.gfile.GFile(path, mode='rb') as f:
        # init GraphDef()
        restored_graph_def = tf.GraphDef()
        # parse saved .pb to GraphDef()
        restored_graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as g_combined:
        # join pipeline to main graph
        pred = tf.import_graph_def(
            graph_def=restored_graph_def,
            input_map={
                'input_pipeline/input_cond/Merge_1': tf.convert_to_tensor(pipeline),
            },
            return_elements=['model/decoder/deconv8bisbis/relu:0'],
            name=''  #note: '' so that won't have import/ prefix
        )


        # prepare feed_dict for inference
        ops_dict = {
            'pred': g_combined.get_tensor_by_name('model/decoder/deconv8bisbis/relu:0'),
        }

        return g_combined, ops_dict


def reconstruct(stack, image_size, step):
    """
    inputs:
    -------
        stack: (np.ndarray) stack of patches to reconstruct
        image_size: (tuple | list) height and width for the final reconstructed image
        step: (int) herein should be the SAME stride step that one used for preprocess
    return:
    -------
        img: (np.ndarray) final reconstructed image
        nb_patches: (int) number of patches need to provide to this function
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = stack.shape[1:3]
    img = np.zeros(image_size)

    # compute the dimensions of the patches array
    n_h = (i_h - p_h) // step + 1
    n_w = (i_w - p_w) // step + 1
    nb_patches = n_h * n_w

    for p, (i, j) in zip(stack, product(range(n_h), range(n_w))):
        img[i * step:i * step + p_h, j * step:j * step + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            img[i, j] /= float(min(i + step, p_h, i_h - i) *
                               min(j + step, p_w, i_w - j))
    return img, nb_patches


def predict_ph(patch_size, batch_size, list_fname, res_dir, pb_path='./dummy/pb/test.pb'):
    """
    inputs:
    -------
        patch_size: (int)
        batch_size: (int)
        predict_dir: (str)
        res_dir: (str)
    return:
    -------
        None
    """
    # determine nodes to conserve and new
    new_ph = tf.placeholder(tf.float32, shape=[batch_size, patch_size, patch_size, 1], name='new_ph')
    conserve_nodes = [
                'model/decoder/deconv8bisbis/relu',
            ]

    # freeze graph to pb
    convert_ckpt2pb(input=new_ph, conserve_nodes=conserve_nodes, pb_path=pb_path)

    # load graph
    g_main, ops_dict = load_mainGraph(conserve_nodes, path=pb_path)

    # run node and stack results
    with g_main.as_default() as g_main:
        new_input = g_main.get_tensor_by_name('new_ph:0')
        dropout_input = g_main.get_tensor_by_name('input_pipeline/dropout_prob:0')

        with tf.Session(graph=g_main) as sess:
            for i, img_fname in tqdm(enumerate(list_fname), desc='nth image prediction'):
                img = np.array(Image.open(img_fname))

                # stride img to np ndarray
                patches = _stride(img, 1, patch_size)
                feed_dict = {
                    new_input: patches[:batch_size].reshape(batch_size, patch_size, patch_size, 1),
                    dropout_input: 1.0,
                }
                res = sess.run(ops_dict['ops'], feed_dict=feed_dict)[0]  # list of result from ops: [(200, 72, 72, 1)]

                for j in range(1, patches.shape[0] // batch_size + 1):  #fixme: last batch don't drop the rest
                    # run inference
                    try:
                        feed_dict = {
                            new_input: patches[j * batch_size: (j + 1) * batch_size].reshape(batch_size, patch_size, patch_size, 1),
                            dropout_input: 1.0,
                        }
                        res = np.vstack((sess.run(ops_dict['ops'], feed_dict=feed_dict)[0], res))
                    except Exception as e:
                        print(e)
                        print('\nlast batch')
                        tmp = patches[j * batch_size:]
                        tmp = np.vstack((np.zeros((batch_size - tmp.shape[0], patch_size, patch_size), 1), np.array(tmp)))
                        feed_dict = {
                            new_input: tmp.reshape(batch_size, patch_size, patch_size, 1),
                            dropout_input: 1.0,
                        }
                        res = np.vstack((sess.run(ops_dict['ops'], feed_dict=feed_dict)[0][tmp.shape[0]:], res[0]))

                # reconstruct to tiff
                recon, _ = reconstruct(res, img.shape, step=1)

                # write tiff
                Image.fromarray(recon).save(res_dir + 'predict_{}.tif'.format(i))


def predict_tfDataset(patch_size, batch_size, list_fname, res_dir, path='./dummy/pb/test.pb'):
    """
    inputs:
    -------
        patch_size: (int)
        batch_size: (int)
        predict_dir: (str)
        res_dir: (str)
    return:
    -------
        None
    """
    # determine nodes to conserve and new
    inputs = inputpipeline(batch_size, suffix='inference')
    conserve_nodes = [
                'input_pipeline/input_cond/Merge_1',
                'model/decoder/deconv8bisbis/relu',
            ]

    # join new pipeline and model / freeze graph to pb
    convert_ckpt2pb(input=inputs, conserve_nodes=conserve_nodes, pb_path=path)

    # load graph
    g_main, ops_dict = load_mainGraph(conserve_nodes, path=path)

    # join pipeline to main graph
    g_combined, opts = greff_pipeline_to_mainGraph(inputs['batch'], path=path)

    # run node and stack results
    with g_combined.as_default() as g_combined:
        # print nodes name
        print_nodes_name(g_combined)

        # feed dropout ph
        dropout_input = g_combined.get_tensor_by_name('input_pipeline_inference/dropout_prob:0')  #??
        img_fname_ph = g_combined.get_tensor_by_name('input_pipeline_inference/fnames_ph:0')  #??
        patch_size_ph = g_combined.get_tensor_by_name('input_pipeline_inference/patch_size_ph:0')
        tmp = np.array(Image.open(list_fname[0]))
        with tf.Session(graph=g_combined) as sess:
            # init iterator
            sess.run(inputs['iter_init_op'])

            # start inference
            for i, img_fname in tqdm(enumerate(list_fname), desc='nth image prediction'):
                # feed_dict
                feed_dict = {
                    img_fname_ph: img_fname,
                    patch_size_ph: patch_size,
                    dropout_input: 1.0,
                }
                res = sess.run(ops_dict['ops'], feed_dict=feed_dict)[0]  # list of result from ops: [(200, 72, 72, 1)]

                # reconstruct to tiff
                recon, _ = reconstruct(res, tmp.shape, step=1)

                # write tiff
                Image.fromarray(recon).save(res_dir + 'predict_{}.tif'.format(i))


if __name__ == '__main__':
    # # params
    # patch_size = 72
    # batch_size = 300
    #
    # # predict path
    # predict_dir = './proc/{}/test/'.format(patch_size)
    # result_dir = './result/pred/'
    # model_dir = './logs/2019_10_8_bs300_ps72_lr0.0001_cs5_nc80_act_leaky_aug_True/hour9_1st_try_end1epBUG/savedmodel/step2000/'
    # model_path = model_dir + 'saved_model.pb'
    #
    # list_fname = []
    # for dirpath, _, fnames in os.walk(predict_dir):
    #     for fname in fnames:
    #         list_fname.append(os.path.abspath(os.path.join(dirpath, fname)))
    #
    # # predict
    # timeit.timeit(predict_tfDataset(patch_size, batch_size, list_fname, result_dir, mdl_dir=model_dir))
    # timeit.timeit(predict_ph(patch_size, batch_size, list_fname, result_dir, pb_path=model_path))
    pass

