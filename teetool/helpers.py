# support functions

import colorsys
import numpy as np
from numpy.linalg import svd, cond, eig

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

def nearest_spd(A):
    """
    nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
    usage: Ahat = nearestSPD(A)

    From Higham: "The nearest symmetric positive semidefinite matrix in the
    Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
    where H is the symmetric polar factor of B=(A + A')/2."

    http://www.sciencedirect.com/science/article/pii/0024379588902236

    arguments: (input)
    A - square matrix, which will be converted to the nearest Symmetric
    Positive Definite Matrix.

    Arguments: (output)
    Ahat - The matrix chosen as the nearest SPD matrix to A.


    RE-CODED FROM MATLAB nearestSPD
    """

    A = np.mat(A)

    # symmetrize A into B
    B = (A + A.transpose()) / 2.

    # Compute the symmetric polar factor of B. Call it H.
    # Clearly H is itself SPD.
    [_, S_diag, V] = svd(B)
    S = np.diag(S_diag)

    H = V*S*V.transpose()

    # get Ahat in the above formula
    Ahat = (B+H) / 2.

    # ensure symmetry
    Ahat = (Ahat + Ahat.transpose()) / 2.

    # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
    p = 1
    k = 0

    while (p != 0):
        c = cond(Ahat)

        if np.isfinite(c):
            p = 0
        else:
            p = 1

        k = k + 1

        if (p != 0):
            # Ahat failed the conditioning test. It must have been just a hair off,
            # due to floating point trash, so it is simplest now just to
            # tweak by adding a tiny multiple of an identity matrix.
            [aneig, _] = eig(Ahat)
            mineig = np.min(aneig)
            addition = np.abs(-mineig*(k**2) + np.finfo(np.float).eps)
            Ahat = Ahat + addition*np.eye(np.size(Ahat,axis=0))

    return Ahat


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
