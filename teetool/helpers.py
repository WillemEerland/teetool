# support functions

import colorsys
import numpy as np
from numpy.linalg import det, inv, svd, cond, eig
from scipy.spatial import Delaunay

def getDistinctColours(ncolours, colour=None):
    """
    returns N distinct colors using the colourspace.
    spreads equally in hue space, then converts to RGB
    """
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
        list_to_return = list(RGB_tuples)
    else:
        # return this colour only
        list_to_return = []
        for i in range(ncolours):
            list_to_return.append( colour )



    return list_to_return

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

def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """

    #print("{0} {1}".format(np.min(hull), np.max(hull)))

    # if not Delaunay, create
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull, qhull_options='QJ')

    res = (hull.find_simplex(p)>=0)

    res_bool = np.array(res,dtype=bool).reshape(-1,1)

    return res_bool

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def gauss_logp(y, ndim, c, A):
    """
    returns value Gaussian log likelihood
    """

    y = np.mat(y)
    c = np.mat(c)
    A = np.mat(A)

    pL1 = 1. * ndim * np.log(2.*np.pi)
    pL2 = 1. * np.log( det(A) )
    pL3 = 1. * (y-c).transpose()*inv(A)*(y-c)

    pL = - 1. / 2. * ( pL1 + pL2 + pL3 )

    return pL

def gauss(y, ndim, c, A):
    """
    returns value Gaussian
    """

    y = np.mat(y)
    c = np.mat(c)
    A = np.mat(A)

    p1 = 1 / np.sqrt(((2*np.pi)**ndim)*det(A))
    p2 = np.exp(-1/2*(y-c).transpose()*inv(A)*(y-c))

    return (p1*p2)

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

    # if zero, return nan, otherwise return log likelihood
    """
    if py == 0.0:
        pyL = np.nan
    else:
        pyL = np.log(py) - np.log(M)  # division by number of Gaussians
        pyL = float(pyL)  # output is a float
    """


    return pyL

def getMaxOutline(ndim):
    """
    returns default outline based on dimensionality
    """
    defaultOutline = []

    for d in range(ndim):
        defaultOutline.append(np.inf)  # min
        defaultOutline.append(-np.inf)  # max

    return defaultOutline

def getGridFromResolution(outline, resolution):
    """
    return xx, yy, (zz), based on outline and resolution
    """

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

def getDimension(cluster_data):
    """
    returns dimension D of data
    """
    (_, Y) = cluster_data[0]
    (_, D) = Y.shape
    return D

def getMinMax(cluster_data):
    """
    returns tuple (xmin, xmax), to normalise data
    """
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

def normalise_data(cluster_data):
    """
    normalises the x dimension
    """

    # determine minimum maximum
    tuple_min_max = getMinMax(cluster_data)

    cluster_data_norm = []

    for (i, (x, Y)) in enumerate(cluster_data):
        xnorm = getNorm(x, tuple_min_max)  # normalise
        # cluster_data[i] = (x, Y)  # overwrite
        cluster_data_norm.append((xnorm, Y))

    return cluster_data_norm

def getNorm(x, tuple_min_max):
    """
    returns normalised array
    """
    (xmin, xmax) = tuple_min_max
    return ((x - xmin) / (xmax - xmin))
