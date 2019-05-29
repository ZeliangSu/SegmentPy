import tensorflow as tf
from visualize import convert_ckpt2pb, load_mainGraph
from proc import _stride
import numpy as np
from itertools import product
from PIL import Image
from tqdm import tqdm
import os


def reconstruct(stack, image_size, step):
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

# params
patch_size = 72
batch_size = 200

# predict path
predict_dir = './raw'
result_dir = './result/pred/'
list_fname = []
for dirpath, _, fnames in os.walk(predict_dir):
    for fname in fnames:
        list_fname.append(os.path.abspath(os.path.join(dirpath, fname)))

# determine nodes to conserve and new
new_ph = tf.placeholder(tf.float32, shape=[batch_size, patch_size, patch_size, 1], name='new_ph')
conserve_nodes = [
            'model/decoder/deconv8bisbis/relu',
        ]
pb_path ='./dummy/pb/test.pb'

# freeze graph to pb
convert_ckpt2pb(input=new_ph, conserve_nodes=conserve_nodes, pb_path=pb_path)

# load graph
g_main, ops_dict = load_mainGraph(conserve_nodes, path=pb_path)

# stride img to np ndarray


# run node and stack results
with g_main.as_default() as g_main:
    new_input = g_main.get_tensor_by_name('new_ph:0')
    dropout_input = g_main.get_tensor_by_name('input_pipeline/dropout_prob:0')

    with tf.Session(graph=g_main) as sess:
        for i, img_fname in tqdm(enumerate(list_fname), desc='nth image prediction'):
            img = np.array(Image.open(img_fname))
            patches = _stride(img, 1, patch_size)
            feed_dict = {
                new_input: patches[0].reshape(batch_size, patch_size, patch_size, 1),
                dropout_input: 1.0,
            }
            res = sess.run([ops_dict['ops']], feed_dict=feed_dict)

            for j in range(1, patches.shape[0]):  #fixme: last batch don't drop the rest
                # run inference
                feed_dict = {
                    new_input: patches[j].reshape(batch_size, patch_size, patch_size, 1),
                    dropout_input: 1.0,
                }
                res = np.vstack((sess.run([ops_dict['ops']], feed_dict=feed_dict), res))

            # reconstruct to tiff
            recon = reconstruct(res, img.shape, step=1)

            # write tiff
            Image.fromarray(recon).save(result_dir + 'predict_{}.tif'.format(i))

