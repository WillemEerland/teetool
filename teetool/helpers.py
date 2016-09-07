# support functions

import colorsys
import numpy as np

def getDistinctColours(ncolours):
    """
    returns N distinct colors using the colourspace.
    spreads equally in hue space, then converts to RGB
    """
    # check
    if type(ncolours) is not int:
        raise TypeError("expected integer, not {0}".format(type(ncolours)))

    if (ncolours < 1):
        raise ValueError("expected integer to be larger than 0, not {0}".format(ncolours))

    # spread equally in hue space
    HSV_tuples = [(x*1.0/ncolours, 0.5, 0.5) for x in range(ncolours)]

    # conver to RGB
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

    return list(RGB_tuples)

def find_nearest(target_array, target_values):
    """
    function to find nearest values in an array, perfect for
    reducing the number of datapoints
    """

    # all array
    target_array = np.array(target_array)
    target_values = np.array(target_values)

    idx = []

    for (i, target_value) in enumerate(target_values):
        # find index
        idx.append((np.abs(target_array-target_value)).argmin())

    return idx

def get_trajectories(ntype=0, ndim=3, ntraj=50, npoints=100, noise_std=.5):
    """
    ntype: different output
    ndim: number of dimensions (2d or 3d)
    ntraj: number of trajectories
    npoints: number of datapoints
    returns a list of trajectories (x, Y)
    """
    # remove random effect
    np.random.seed(seed=10)

    # generate toy trajectories

    x = np.linspace(-50, 50, num=npoints)

    toy_trajectories = []

    for i in range(ntraj):

        if (ntype == 0):
            # [first set of trajectories]
            y1 = x + 2*np.random.randn(1) - 2.5
            y2 = 0.05*(x**2) + 10*np.random.randn(1) + 90
            y3 = 0.03*(x**2) + 3*np.random.randn(1) + 2.5
        else:
            # [second set of trajectories]
            y1 = x + 2*np.random.randn(1) - 2.5
            y2 = -x + 5*np.random.randn(1) + 45
            y3 = -0.03*(x**2) + 3*np.random.randn(1) + 2.5

        # 2d / 3d
        if (ndim == 2):
            Y = np.array([y1, y2]).transpose()

        if (ndim == 3):
            Y = np.array([y1, y2, y3]).transpose()

        # add noise
        (nrows, ncols) = Y.shape
        Y += noise_std*np.random.randn(nrows, ncols)

        toy_trajectories.append( (x, Y) )

    return toy_trajectories
