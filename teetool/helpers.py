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

def get_trajectories(ntype=0, ndim=3, ntraj=50, npoints=100):
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
            y1 = x + 5*np.random.rand(1) - 2.5
            y2 = 0.05*(x**2) + 20*np.random.rand(1) + 80
            y3 = .3*x + 5*np.random.rand(1) - 7
        else:
            # [second set of trajectories]
            y1 = x + 5*np.random.rand(1) - 2.5
            y2 = -x + 20*np.random.rand(1) + 50
            y3 = -0.03*(x**2) + 3*np.random.rand(1) + 2.5

        # 2d / 3d
        if (ndim == 2):
            Y = np.array([y1, y2]).transpose()

        if (ndim == 3):
            Y = np.array([y1, y2, y3]).transpose()

        toy_trajectories.append( (x, Y) )

    return toy_trajectories

def block_diag(*arrs):
    """Create a block diagonal matrix from the provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Parameters
    ----------
    A, B, C, ... : array-like, up to 2D
        Input arrays.  A 1D array or array-like sequence with length n is
        treated as a 2D array with shape (1,n).

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
        same dtype as `A`.

    References
    ----------
    .. [1] Wikipedia, "Block matrix",
           http://en.wikipedia.org/wiki/Block_diagonal_matrix

    Examples
    --------
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> print(block_diag(A, B, C))
    [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 3 4 5 0]
     [0 0 6 7 8 0]
     [0 0 0 0 0 7]]
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out
