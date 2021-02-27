import numpy as np
import utils
import matplotlib.pyplot as plt
def add(img, alpha):
    #adds alpha to all pixels of an image.
    #Additionally clips values to between 0 and 1 (see utils.clip)
    # TODO 1a
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END

def multiply(img, alpha):
    # multiplies all pixel intensities by alpha
    # additionally clips values
    # TODO 1b
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END

def normalize(img):
    # Performs an affine transformation of the intensities
    # (i.e., f'(x,y) = af(x,y) + b) with a and b chosen
    # so that the minimum value is 0 and the maximum value is 1.
    # Input: w x h grayscale image of dtype np.float
    # Output: w x h  grayscale image of dtype np.float with minimum value 0
    # and maximum value 1
    # If all pixels in the image have the same intensity,
    # then return an image that is 0 everywhere
    # TODO 1c
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END
def threshold(img, thresh):
    # Produces an image where pixels greater than thresh are assigned 1 and
    # those less than thresh are assigned 0
    # Make sure to return a float image
    # TODO 1d
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END


def convolve(img, filt):
    # Performs a convolution of an image with a filter.
    # Assume filter is 2D and has an odd size
    # Assume image is grayscale
    # Perform "same" convolution, i.e., output image should be the ssame size
    # as the input
    assert len(img.shape)==2, "Image must be grayscale."
    assert len(filt.shape)==2, "Filter must be 2D."
    assert ((filt.shape[0]%2!=0) and (filt.shape[1]%2!=0)), "Filter dimensions must be odd."

    # TODO 2
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END

def mean_filter(k):
    # Produces a k x k mean filter
    # Assume k is odd
    assert k%2!=0, "Kernel size must be odd"
    # TODO 3a
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END


def gaussian_filter(k, sigma):
    # Produces a k x k gaussian filter with standard deviation sigma
    # Assume k is odd
    assert k%2!=0, "Kernel size must be odd"

    # TODO 3b
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END

def dx_filter():
    # Produces a 1 x 3 filter that computes the derivative in x direction
    # TODO 4a
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END

def dy_filter():
    # Produces a 3 x 1 filter that computes the derivative in y direction
    # TODO 4b
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END

def gradient_magnitude(img, k=3,sigma=0):
    # Compute the gradient magnitude at each pixel,,
    # If sigma >0, smmooth the
    # image with a gaussian first
    # TODO 4c
    # TODO-BLOCK-BEGIN
    pass
    # TODO-BLOCK-END
