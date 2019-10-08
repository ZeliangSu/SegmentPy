import tensorflow as tf
import numpy as np
from random import choice


def random_aug(X_img, y_img):
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
    fns = [gaussian_noise, flipping, sp_noise, speckle_noise, non_noise]  #todo: can add probabilities
    X_img, y_img = choice(fns)(X_img, y_img)
    return _minmaxscalar(X_img), y_img


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
    return _minmaxscalar(X_img), y_img


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
    return _minmaxscalar(X_img), y_img


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
    X_img[coords] = salt
    # pepper
    coords = [np.random.randint(0, i, nb) for i in X_img.shape]
    X_img[coords] = pepper
    return _minmaxscalar(X_img), y_img


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
    return _minmaxscalar(X_img), y_img


def non_noise(X_img, y_img):
    '''Do nothing'''
    return _minmaxscalar(X_img), y_img


def poisson_noise():
    raise NotImplementedError('No poisson noise yet!')


def contrast():
    raise NotImplementedError('No variant intensity augmentation yet!')


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