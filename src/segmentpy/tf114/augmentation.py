import numpy as np
from random import choice
from scipy import ndimage, interpolate

# logging
import logging
from segmentpy.tf114 import log
logger = log.setup_custom_logger(__name__)
logger.setLevel(logging.INFO)  #changeHere: debug level


def random_aug(X_img,
               y_img,  # 1e-4 if range of input  (0, 0.0001)
               warp=True,
               ):
    """
    input:
    -------
    X_img: (np.ndarray) image to augment
    y_img: (np.ndarray) image to augment

    return:
    -------
    X_img: (np.ndarray) augmented input image
    y_img: (np.ndarray) same output image
    """
    fns = [
        gaussian_noise,
        flipping,
        sp_noise,
        #speckle_noise,
        non_noise,
        # contrast,
        grayscale_var,
    ]  #todo: can add probabilities
    if warp:
        fns.append(warping)

    # X_img = X_img.copy() * normalization
    X_img, y_img = choice(fns)(X_img, y_img)
    y_img.astype(np.int32)

    #note: minmaxscal will alternate if not all classes are present
    # return _minmaxscalar(X_img).astype('float32'), y_img

    return X_img.astype('float32'), y_img


def gaussian_noise(X_img, y_img, sigma=0.1):
    """
    input:
    -------
    X_img: (np.ndarray) image to augment
    y_img: (np.ndarray) image to augment
    sigma: (float) normal distribution parameter

    return:
    -------
    X_img: (np.ndarray) gaussian noise added input image
    y_img: (np.ndarray) same output image
    """
    mu = np.mean(X_img)
    noise = np.random.normal(mu, sigma, size=X_img.shape)
    X_img += noise
    return X_img, y_img


def flipping(X_img, y_img):
    """
    input:
    -------
        X_img: (np.ndarray) image to augment
        y_img: (np.ndarray) image to augment

    return:
    -------
        X_img: (np.ndarray) flipped input image
        y_img: (np.ndarray) same output image
    """
    choice = np.random.choice(['ud', 'lr'])
    if choice == np.array(['ud']):
        X_img, y_img = np.flipud(X_img), np.flipud(y_img)
    elif choice == np.array(['lr']):
        X_img, y_img = np.fliplr(X_img), np.fliplr(y_img)
    else:
        raise ValueError('please choose between "ud" as up-down or "lr" as left-right!')
    return X_img, y_img


def sp_noise(X_img, y_img, amount=0.005):
    """
    input:
    -------
        X_img: (np.ndarray) image to augment
        y_img: (np.ndarray) image to augment
        amount: (float) probability of adding an S&P noise

    return:
    -------
        X_img: (np.ndarray) salt&pepper noise added input image
        y_img: (np.ndarray) same output image
    """
    salt = np.max(X_img)
    pepper = np.min(X_img)
    nb = int(amount * X_img.size * 0.5)
    # salt
    coords = [np.random.randint(0, i, nb) for i in X_img.shape]
    X_img[tuple(coords)] = salt
    # pepper
    coords = [np.random.randint(0, i, nb) for i in X_img.shape]
    X_img[tuple(coords)] = pepper
    return X_img, y_img


def speckle_noise(X_img, y_img):
    """
    input:
    -------
        X_img: (np.ndarray) image to augment
        y_img: (np.ndarray) image to augment

    return:
    -------
        X_img: (np.ndarray) speckle noise added input image
        y_img: (np.ndarray) same output image
    """
    weighting = np.random.randn(*X_img.shape)
    X_img = X_img + X_img * weighting
    return X_img, y_img


def non_noise(X_img, y_img):
    '''Do nothing'''

    return X_img, y_img


def grayscale_var(X_img, y_img):
    std = X_img.std()
    X_img += std * np.random.uniform(-1, 1)
    return X_img, y_img


def contrast(X_img, y_img):
    min = X_img.min()
    rand = np.random.uniform(0.2, 1)
    X_img = (X_img - min) * rand + min
    return X_img, y_img


def warping(X_img, y_img):
    # y_img.shape == [H, W, nb_cls] or [H, W]
    logger.debug('X shape: {}, y shape: {}'.format(X_img.shape, y_img.shape))
    assert X_img.shape[:2] == y_img.shape[:2]

    rows, cols = np.meshgrid(range(X_img.shape[1]), range(X_img.shape[0]))
    rows = rows ** (1 / 2) * (X_img.shape[1] - 1) ** (1 / 2)  #todo: make this random?
    cols = cols ** (2) / (X_img.shape[0] - 1)  #todo: make this random?

    if len(X_img.shape) == 3:
        for i in range(X_img.shape[2]):
            X_img[:, :, i] = ndimage.map_coordinates(X_img[:, :, i], [cols, rows], order=3)
    else:
        X_img = ndimage.map_coordinates(X_img, [cols, rows], order=3)

    if len(y_img.shape) == 3:
        y_img = y_img.copy()
        for i in range(y_img.shape[2]):
            y_img[:, :, i] = ndimage.map_coordinates(y_img[:, :, i], [cols, rows], order=3)
    else:
        y_img = ndimage.map_coordinates(y_img, [cols, rows], order=3)
    return X_img, y_img


def _minmaxscalar(ndarray, dtype=np.float32):
    """
    func normalize values of a ndarray into interval of 0 to 1

    input:
    -------
        ndarray: (numpy ndarray) input array to be normalized
        dtype: (dtype of numpy) data type of the output of this function

    output:
    -------
        scaled: (numpy ndarray) output normalized array
    """
    scaled = np.array((ndarray - np.min(ndarray)) / (np.max(ndarray) - np.min(ndarray)), dtype=dtype)
    return scaled


