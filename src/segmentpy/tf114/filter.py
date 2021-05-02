import numpy as np
import multiprocessing as mp
import cv2

#todo: check filter with WEKA
ddepth = cv2.CV_16S


def wrapper(f, *args):
    print(mp.current_process())  # see process ID
    return f(*args)


def Gaussian_Blur(img):
    out = cv2.GaussianBlur(img, (5, 5), 0)
    return out


def Hessian(img):
    '''https://stackoverflow.com/questions/48727914/how-to-use-ridge-detection-filter-in-opencv'''
    out = cv2.ximgproc.RidgeDetectionFilter_create().getRidgeFilteredImage(img)
    return out


def DoG(img):
    '''https://answers.opencv.org/question/187563/difference-of-gaussian-different-outputs-in-opencv-and-imagemagick/'''
    g1 = cv2.getGaussianKernel(ddepth, 5)
    g2 = cv2.getGaussianKernel(ddepth, 0)
    kernel = g1 - g2
    out = cv2.filter2D(img, -1, kernel)
    return out


def Laplacian(img):
    out = cv2.Laplacian(img, ddepth=ddepth, ksize=5)
    return out


def Anisotropic_Diffusion1(img):
    '''https://pastebin.com/sBsPX4Y7'''
    gamma = 0.1
    step = (1., 1.)
    kappa = 50
    deltaS = np.zeros_like(img)
    deltaE = deltaS
    NS = EW = deltaS
    out = img.copy()
    for ii in range(1):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(out, axis=0)
        deltaE[:, :-1] = np.diff(out, axis=1)

        # note: here might have an effect
        # Diffusion equation 1 favours high contrast edges over low contrast ones.
        # Diffusion equation 2 favours wide regions over smaller ones

        # diff eq 1
        gS = np.exp(-(deltaS / kappa) ** 2.) / step[0]
        gE = np.exp(-(deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        out += gamma * (NS + EW)
    return out


def Anisotropic_Diffusion2(img):
    '''https://pastebin.com/sBsPX4Y7'''
    gamma = 0.1
    step = (1., 1.)
    kappa = 50
    deltaS = np.zeros_like(img)
    deltaE = deltaS
    NS = EW = deltaS
    out = img.copy()
    for ii in range(1):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(img, axis=0)
        deltaE[:, :-1] = np.diff(img, axis=1)

        # note: here might have an effect
        # Diffusion equation 1 favours high contrast edges over low contrast ones.
        # Diffusion equation 2 favours wide regions over smaller ones

        # diff eq 2
        gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[0]
        gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        out += gamma * (NS + EW)
    return out


def Bilateral(img):
    out = cv2.bilateralFilter(img, 9, 75, 75)
    return out


def Median(img):
    out = cv2.medianBlur(img, 5)
    return out


def Gabor(img):
    '''https://gist.github.com/kendricktan/93f0da88d0b25087d751ed2244cf770c'''
    kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
    out = cv2.filter2D(img, ddepth=ddepth, kernel=kernel)
    return out


def Sobel(img):
    '''https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html'''
    out = cv2.Sobel(img, ddepth=ddepth, dx=1, dy=1)
    return out


def Membrane_proj(img):
    raise NotImplementedError('Membrane projection not implemented')


def structure(img):
    raise NotImplementedError('Structure filter not implemented')


def Entropy(img):
    raise NotImplementedError('Entropy filter not implemented')


def Variance(img):
    raise NotImplementedError('Varince filter not implemented')


def Unsharp(img):
    raise NotImplementedError('Unsharp not implemented')
