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

def get_trajectories(ntype=0, D=3, N=50):
    """
    ntype: different output
    d: number of dimensions (2d or 3d)
    returns a list of trajectories (x, Y)
    """
    # remove random effect
    np.random.seed(seed=10)

    # PARAMETERS
    M = 100  # number of data-points per trajectory

    # generate toy trajectories

    x = np.linspace(-50, 50, num=M)

    toy_trajectories = []

    for i in range(N):

        if (ntype == 0):
            # [first set of trajectories]
            y1 = x + 5*np.random.rand(1) - 2.5
            y2 = 0.05*(x**2) + 20*np.random.rand(1) + 80
            y3 = .3*x + 5*np.random.rand(1) - 7
        else:
            # [second set of trajectories]
            y1 = x + 5*np.random.rand(1) - 2.5
            y2 = -x + 20*np.random.rand(1) + 50
            y3 = -0.03*(x**2) + 3*np.random.rand(1) + 2.5

        # 2d / 3d
        if (D == 2):
            Y = np.array([y1, y2]).transpose()

        if (D == 3):
            Y = np.array([y1, y2, y3]).transpose()

        toy_trajectories.append( (x, Y) )

    return toy_trajectories
