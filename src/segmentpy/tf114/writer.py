import numpy as np
from PIL import Image
from segmentpy.tf114.util import check_N_mkdir, clean, auto_contrast

import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger('root')
logger.setLevel(logging.DEBUG)


def _resultWriter(tensor, layer_name='', path=None, batch_or_channel='batch', contrast=True):
    '''
    tensor: images(numpy array or list of image) to save of (Height, Width, nth_Conv)
    path: path(string)
    layer_name: name of the layer(string)
    '''
    # mkdir
    check_N_mkdir(path + layer_name)

    # for writting inference partial rlt
    if isinstance(tensor, list):
        for i, elt in enumerate(tensor):
            logger.debug('layer_name: {}'.format(layer_name))
            if (True in np.isnan(elt)) or (True in np.isinf(elt)):
                try:
                    elt = clean(elt)
                except Exception as e:
                    logger.debug(e)
                    logger.warning('Clean function cannot clean the nan value from the layer {}! '.format(layer_name))
                    break

            # scope: conv p_infer: (ps, ps, nc)
            if elt.ndim == 3:
                # note: save only the first
                for j in range(elt.shape[-1]):
                    activation = np.asarray(elt[:, :, j])
                    if contrast:
                        activation = auto_contrast(activation)
                    Image.fromarray(activation).save(path + '{}/{}.tif'.format(layer_name, j))

            # scope: images
            elif elt.ndim == 2:
                Image.fromarray(np.asarray(elt)).save(path + '{}/{}.tif'.format(layer_name, i))

            # scope: dnn p_infer: (nb_post neuron)
            elif elt.ndim == 1:
                # reshape the 1D array
                # note: round the shape into a square
                ceil = int(np.ceil(np.sqrt(elt.size)))
                tmp = np.zeros((ceil ** 2), np.float32).ravel()
                tmp[:elt.size] = elt
                tmp = tmp.reshape(ceil, ceil)

                Image.fromarray(tmp).save(path + '{}/dnn.tif'.format(layer_name))

            elif elt.ndim == 4:
                elt = elt.squeeze()
                for j in range(elt.shape[0]):
                    activation = np.asarray(elt[j])
                    if contrast:
                        activation = auto_contrast(activation)
                    Image.fromarray(activation).save(path + '{}/{}.tif'.format(layer_name, j))
    else:
        #  treat dnn weights
        if tensor.ndim == 1:
            # reshape the 1D array
            ceil = int(np.ceil(np.sqrt(tensor.size)))
            tmp = np.zeros((ceil ** 2), np.float32).ravel()
            tmp[:tensor.size] = tensor
            tmp = tmp.reshape(ceil, ceil)

            Image.fromarray(tmp).save(path + '{}/dnn.tif'.format(layer_name))

        #  scope: not I_1_hoted tensor: (B, H, W, C)
        elif tensor.ndim == 4:
            tensor = np.squeeze(tensor.astype(np.float32))
            if 'diff' in layer_name or 'logit' in layer_name:
                if tensor.ndim == 2:
                    Image.fromarray(tensor).save(path + '{}/{}.tif'.format(layer_name, 0))
                elif batch_or_channel == 'batch':
                    for i in range(tensor.shape[0]):
                        Image.fromarray(tensor[i]).save(path + '{}/{}.tif'.format(layer_name, i))
                elif batch_or_channel == 'channel':
                    for i in range(tensor.shape[-1]):
                        Image.fromarray(tensor[i]).save(path + '{}/{}.tif'.format(layer_name, i))
                else:
                    raise NotImplementedError

        #  for cnn ndim=3
        elif tensor.ndim == 3:
            for i in range(tensor.shape[2]):
                activation = np.asarray(tensor[:, :, i], dtype=np.float)
                if contrast:
                    activation = auto_contrast(activation)
                Image.fromarray(activation).save(path + '{}/{}.tif'.format(layer_name, i))

        else:
            logger.warn('Not implement writer for this kind of data in layer :{}'.format(layer_name))
            raise NotImplementedError('Ouppss {}'.format(layer_name))


def _weighttWriter(tensor, layer_name='', path=None):
    '''
    tensor: images(numpy array or list of image) to save of (Height, Width, nth_Conv)
    path: path(string)
    layer_name: name of the layer(string)
    '''
    # mkdir
    check_N_mkdir(path + layer_name)

    # for writting inference partial rlt
    if isinstance(tensor, list):
        for i, elt in enumerate(tensor):
            logger.debug('layer_name: {}'.format(layer_name))
            if (True in np.isnan(elt)) or (True in np.isinf(elt)):
                try:
                    elt = clean(elt)
                except Exception as e:
                    logger.debug(e)
                    logger.warning('Clean function cannot clean the nan value from the layer {}! '.format(layer_name))
                    break

            # scope: conv p_infer: (ps, ps, nc)
            if elt.ndim == 3:
                # note: save only the first
                for j in range(elt.shape[-1]):
                    Image.fromarray(np.asarray(elt[:, :, j])).save(path + '{}/{}.tif'.format(layer_name, j))

            elif elt.ndim == 2:
                Image.fromarray(np.asarray(elt)).save(path + '{}/{}.tif'.format(layer_name, i))

            # scope: dnn p_infer: (nb_post neuron)
            elif elt.ndim == 1:
                # reshape the 1D array
                ceil = int(np.ceil(np.sqrt(elt.size)))
                tmp = np.zeros((ceil ** 2), np.float32).ravel()
                tmp[:elt.size] = elt
                tmp = tmp.reshape(ceil, ceil)

                Image.fromarray(tmp).save(path + '{}/dnn.tif'.format(layer_name))

    else:
        #  treat dnn weights
        if tensor.ndim == 1:
            # reshape the 1D array
            ceil = int(np.ceil(np.sqrt(tensor.size)))
            tmp = np.zeros((ceil ** 2), np.float32).ravel()
            tmp[:tensor.size] = tensor
            tmp = tmp.reshape(ceil, ceil)

            Image.fromarray(tmp).save(path + '{}/dnn.tif'.format(layer_name))

        #  scope: not I_1_hoted tensor: (B, H, W, C)
        elif tensor.ndim == 4:
            tensor = np.squeeze(tensor.astype(np.float32))
            if 'diff' in layer_name:
                for i in range(tensor.shape[0]):
                    Image.fromarray(tensor[i]).save(path + '{}/{}.tif'.format(layer_name, i))

        #  for cnn ndim=3
        elif tensor.ndim == 3:
            for i in range(tensor.shape[2]):
                Image.fromarray(np.asarray(tensor[:, :, i], dtype=np.float)).save(path + '{}/{}.tif'.format(layer_name, i))

        else:
            logger.warn('Not implement writer for this kind of data in layer :{}'.format(layer_name))
            raise NotImplementedError('Ouppss {}'.format(layer_name))
