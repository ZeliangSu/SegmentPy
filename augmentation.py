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
    fns = [gaussian_noise, flipping, sp_noise, speckle_noise]  #todo: can add probabilities
    X_img, y_img = choice(fns)(X_img, y_img)
    return X_img, y_img


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
    nb = np.ceil(amount * X_img.size * 0.5)
    # salt
    coords = [np.random.randint(0, i - 1, nb) for i in X_img.shape]
    X_img[coords] = salt
    # pepper
    coords = [np.random.randint(0, i - 1, nb) for i in X_img.shape]
    X_img[coords] = pepper
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
    weighting = np.random.randn(X_img.shape)
    X_img = X_img + X_img * weighting
    return X_img, y_img


def poisson_noise():
    raise NotImplementedError('No poisson noise yet!')


def contrast():
    raise NotImplementedError('No variant intensity augmentation yet!')