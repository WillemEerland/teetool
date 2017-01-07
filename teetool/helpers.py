## @package teetool
#  This module contains the support functions used in the teetool package
#
#  All these functions are used in multiple classes, hence it was better to create a single point for all these 'general' functions -- some of these are also used for examples

import colorsys
import numpy as np
from numpy.linalg import det, inv, svd, cond, eig
from scipy.spatial import Delaunay

import teetool as tt

## a function to obtain n distinct colours (based on the colourspace spread into hue space), inspired by https://stackoverflow.com/questions/470690/how-to-automatically-generate-n-distinct-colors
# @param ncolours integer with number of distinct colours
# @param colour when colour is specified, this overrides any distinct colour
# @return rgb_tuples a list of tuples with (R, G, B) in the [0,1] domain
def getDistinctColours(ncolours, colour=None):
    # check
    if type(ncolours) is not int:
        raise TypeError("expected integer, not {0}".format(type(ncolours)))

    if (ncolours < 1):
        raise ValueError("expected integer to be larger than 0, not {0}".format(ncolours))

    if colour is None:
        # spread equally in hue space
        HSV_tuples = [(x*1.0/ncolours, 0.5, 0.5) for x in range(ncolours)]
        # convert to RGB
        RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
        # convert to list
        return list(RGB_tuples)
    else:
        # return this colour only
        list_to_return = []
        for i in range(ncolours):
            list_to_return.append( colour )
        return list_to_return

## function to find nearest values in an array, perfect for reducing the number of datapoints
# @param original_values an array of original values
# @param target_values an array of desired values
# @return idx returns a list of indices of points that hold the nearest values
def find_nearest(original_values, target_values):
    # all array
    original_values = np.array(original_values)
    target_values = np.array(target_values)

    idx = []

    for (i, target_value) in enumerate(target_values):
        # find index
        idx.append((np.abs(original_values-target_value)).argmin())

    return idx


## nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
# @param A input matrix
# @return Ahat nearest SPD matrix to A
#
# nearestSPD - the nearest (in Frobenius norm) Symmetric Positive Definite matrix to A
# usage: Ahat = nearestSPD(A)
#
# From Higham: "The nearest symmetric positive semidefinite matrix in the
# Frobenius norm to an arbitrary real matrix A is shown to be (B + H)/2,
# where H is the symmetric polar factor of B=(A + A')/2."
#
# http://www.sciencedirect.com/science/article/pii/0024379588902236
#
# arguments: (input)
# A - square matrix, which will be converted to the nearest Symmetric
# Positive Definite Matrix.
#
# Arguments: (output)
# Ahat - The matrix chosen as the nearest SPD matrix to A.
#
# RE-CODED FROM MATLAB nearestSPD
# http://uk.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
def nearest_spd(A):
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

## function to generate some distributed trajectories
# @param ntype type of trajectory (0 or 1)
# @param ndim dimensionality of trajectories (2 or 3)
# @param ntraj number of trajectories desired
# @param npoints how many points each trajectory has
# @param noise_std includes normal distributed (Gaussian) noise to the trajectory data-points
# return cluster_data a list of (x, Y) trajectory data
def get_trajectories(ntype=0, ndim=3, ntraj=50, npoints=100, noise_std=.5):
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

## Test if points in `p` are in `hull`
# @param p an array of points
# @param hull points for hull, if not hull, make a hull from points
# @return p_bool an array with bools whether or not the points are inside the hull
#
# `p` should be a `NxK` coordinates of `N` points in `K` dimensions `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the coordinates of `M` points in `K`dimensions for which Delaunay triangulation will be computed
#
# https://stackoverflow.com/questions/16750618/whats-an-efficient-way-to-find-if-a-point-lies-in-the-convex-hull-of-a-point-cl
def in_hull(p, hull):
    # if not Delaunay, create
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull, qhull_options='QJ')

    res = (hull.find_simplex(p)>=0)

    res_bool = np.array(res,dtype=bool).reshape(-1,1)

    return res_bool

## function to find unique rows
# @param a matrix
# @return unique_A the matrix a with only unique rows
#
# https://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

## function to evaluate the Gaussian log likelihood
# @param y data-point
# @param ndim dimensionality of y
# @param c mean vector
# @param A convariance matrix
# @returns loglikelihood log likelihood of Gaussian
def gauss_logp(y, ndim, c, A):
    y = np.mat(y)
    c = np.mat(c)
    A = np.mat(A)

    pL1 = 1. * ndim * np.log(2.*np.pi)
    pL2 = 1. * np.log( det(A) )
    pL3 = 1. * (y-c).transpose()*inv(A)*(y-c)

    pL = - 1. / 2. * ( pL1 + pL2 + pL3 )

    return pL

## function to evaluate the Gaussian (normal distribution)
# @param y data-point
# @param ndim dimensionality of y
# @param c mean vector
# @param A convariance matrix
# @returns value value of Gaussian at point y
def gauss(y, ndim, c, A):
    y = np.mat(y)
    c = np.mat(c)
    A = np.mat(A)

    p1 = 1 / np.sqrt(((2*np.pi)**ndim)*det(A))
    p2 = np.exp(-1/2*(y-c).transpose()*inv(A)*(y-c))

    return (p1*p2)

## function to evaluate the Gaussian log likelihood in cell-form
# @param y data-point
# @param ndim dimensionality of y
# @param cc mean vector cells
# @param cA convariance matrix cells
# @returns loglikelihood maximum log likelihood of Gaussian, based on models found in cells
def gauss_logLc(y, ndim, cc, cA):
    """
    returns the log likelihood of a position based on model (in cells)
    """

    y = y.reshape((ndim, 1), order='F')

    y = np.array(y)

    # check dimension y
    if not y.shape == (ndim, 1):
        raise ValueError("dimension is {0}, not {1}".format(y.shape, (ndim, 1)))

    M = len(cc)

    pyL = - np.inf

    for m in range(M):
        c = cc[m]
        A = cA[m]
        pyLm = gauss_logp(y, ndim, c, A)
        if pyLm > pyL:
            pyL = pyLm
        # py += gauss(y, ndim, c, A)  # addition of each Gaussian

    return pyL

## simple function to obtain infinite outline based on dimensionality
# @param ndim number of dimensions
# @return outline [ndim x 2], for minimum and maximum
def getMaxOutline(ndim):
    defaultOutline = []

    for d in range(ndim):
        defaultOutline.append(np.inf)  # min
        defaultOutline.append(-np.inf)  # max

    return defaultOutline

## function to generate a grid based on outline and resolution
# @param outline [ndim x 2], minimum and maximum values
# @param resolution can be a scalar [1x1] or specified per dimension [ndim x 1]
# @return xx x grid
# @return yy y grid
# @return zz (optional) z grid
def getGridFromResolution(outline, resolution):
    if type(resolution) is not list:
        # create an equal sized grid

        [xmin, xmax, ymin, ymax] = outline[:4]

        xnsteps = int( np.around( (xmax-xmin) / (1.*resolution) ) )
        ynsteps = int( np.around( (ymax-ymin) / (1.*resolution) ) )

        if xnsteps < 2:
            xnsteps = 2

        if ynsteps < 2:
            ynsteps = 2

        if len(outline) is 4:
            # 2d
            [xx, yy] = np.mgrid[xmin:xmax:np.complex(0, xnsteps+1),
                           ymin:ymax:np.complex(0, ynsteps+1)]
            zz = None
        else:
            # 3d
            [zmin, zmax] = outline[4:6]
            znsteps = int( np.around( (zmax-zmin) / (1.*resolution) ) )
            if znsteps < 2:
                znsteps = 2
            [xx, yy, zz] = np.mgrid[xmin:xmax:np.complex(0, xnsteps+1),
                           ymin:ymax:np.complex(0, ynsteps+1),
                           zmin:zmax:np.complex(0, znsteps+1)]

    else:
        # create a grid based on resolution

        [xmin, xmax, ymin, ymax] = outline[:4]

        if len(outline) is 4:
            # 2d
            [xx, yy] = np.mgrid[xmin:xmax:np.complex(0, resolution[0]),
                           ymin:ymax:np.complex(0, resolution[1])]
            zz = None
        else:
            # 3d
            [zmin, zmax] = outline[4:6]
            [xx, yy, zz] = np.mgrid[xmin:xmax:np.complex(0, resolution[0]),
                           ymin:ymax:np.complex(0, resolution[1]),
                           zmin:zmax:np.complex(0, resolution[2])]

    return [xx, yy, zz]

## finds dimension D based on cluster_data
# @param cluster_data list of (x, Y) format
# @return D dimensionality of trajectory data
def getDimension(cluster_data):
    (_, Y) = cluster_data[0]
    (_, D) = Y.shape
    return D

## find minimum/maximum of the cluster data (x, Y), x-component. The tuple is used to normalise the data
# @param cluster_data list of (x, Y)
# @returns xmin minimum value of x
# @returns xmax maximum value of x
def getMinMax(cluster_data):
    xmin = np.inf
    xmax = -np.inf
    for (x, Y) in cluster_data:
        x1min = x.min()
        x1max = x.max()

        if (x1min < xmin):
            xmin = x1min
        if (x1max > xmax):
            xmax = x1max

    return (xmin, xmax)

## normalise the cluster_data in x-domain
# @param cluster_data list of (x, Y) data
# @return cluster_data_norm list of (x, Y) data with normalised x
def normalise_data(cluster_data):
    # determine minimum maximum
    tuple_min_max = getMinMax(cluster_data)

    cluster_data_norm = []

    for (i, (x, Y)) in enumerate(cluster_data):
        xnorm = getNorm(x, tuple_min_max)  # normalise
        # cluster_data[i] = (x, Y)  # overwrite
        cluster_data_norm.append((xnorm, Y))

    return cluster_data_norm

## returns a normalised array based on a specified minimum/maximum
# @param x input array
# @param tuple_min_max a tuple with minimum and maximum specified
# @return x_norm a normalised array x
def getNorm(x, tuple_min_max):
    (xmin, xmax) = tuple_min_max
    return ((x - xmin) / (xmax - xmin))

## returns the outline based on cluster_data
# @param cluster_data list of (x, Y) trajectory data
# @return outline [min, max]*ndim for outline
def get_cluster_data_outline(cluster_data):
    ndim = getDimension(cluster_data)

    this_cluster_data_outline = tt.helpers.getMaxOutline(ndim)

    for (x, Y) in cluster_data:

        for d in range(ndim):
            x = Y[:, d]
            xmin = x.min()
            xmax = x.max()
            if (this_cluster_data_outline[d*2] > xmin):
                this_cluster_data_outline[d*2] = xmin
            if (this_cluster_data_outline[d*2+1] < xmax):
                this_cluster_data_outline[d*2+1] = xmax

    return this_cluster_data_outline

## returns a normalised cluster_data based on outline
# @param cluster_data list of (x, Y) trajectory data
# @param outline array specifying an pre-set outline, otherwise the outline is calculated from the cluster_data
# @return cluster_data_norm list of (x, Y), where all Y and x fit in [0, 1] domain
#
# specifying an outline is useful when wanting to normalise based on multiple clusters
def get_cluster_data_norm(cluster_data, outline=None):
    ndim = getDimension(cluster_data)

    if outline is None:
        outline = get_cluster_data_outline(cluster_data)

    cluster_data_norm = []

    for (t, Y) in cluster_data:

        # Y gets normalised globally
        Y_norm = np.zeros_like(Y)

        for d in range(ndim):
            # cluster outline
            xmin = outline[d*2+0]
            xmax = outline[d*2+1]
            # extract
            x = Y[:, d]
            # normalise
            x_norm = (x - xmin) / (xmax - xmin)
            # place
            Y_norm[:, d] = x_norm

        # t gets normalised locally
        t = np.array(t)
        t_norm = (t - t.min()) / (t.max() - t.min())

        # append to cluster
        cluster_data_norm.append((t_norm, Y_norm))

    return cluster_data_norm
