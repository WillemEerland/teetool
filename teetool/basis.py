## @package teetool
#  This module contains the Basis class
#
#  See Basis class for more details

import numpy as np
from scipy.linalg import block_diag

## Basis class supports the evaluation of a basis function
#
#  These Basis functions function on the domain [0,1], however produce these on multiple dimensions by returning a block-diagonal matrix. Hence an identical basis function gets returned on each dimension.
class Basis(object):

    ## The constructor of Basis
    #   @param self         object pointer
    #   @param basisType    'rbf' or 'bernstein' basis-type where
    #                       'rbf' = uniformly distributed basis functions and
    #                       'bernstein' = Bernstein polynomials
    #   @param nbasis       number of basis functions used
    #   @param ndim         number of dimensions trajectories
    def __init__(self, basisType, nbasis, ndim):
        SUPPORTED_FUNCTIONS = ["rbf", "bernstein"]

        if basisType not in SUPPORTED_FUNCTIONS:
            raise NotImplementedError("{0} type not supported, only {1}".format(basisType, SUPPORTED_FUNCTIONS))

        ## 'rbf' or 'bernstein' basis type
        self._basisType = basisType
        ## number of basis functions
        self._nbasis = nbasis
        ## number of dimensions
        self._ndim = ndim
        margin = 1/(2*nbasis)
        ## defines range of basis function
        self._range = [0+margin, 1-margin]

    ## obtain basis functions at specific points, block-diagonal
    #  @param self  The object pointer.
    #  @param x     points to evaluate [npoints x 1]
    #  @return      values of basis functions [ndim*npoints x ndim*nbasis]
    def get(self, x):
        # check input x
        x_vec = np.array(x)

        mpoints = len(x_vec)
        mbasis = self._nbasis
        mdim = self._ndim

        # get basis for a single dimension
        BASIS_1d = self._get_1d(x_vec)

        # return it for mdim
        BASIS_list = []

        for d in range(mdim):
            BASIS_list.append(BASIS_1d)

        BASIS = block_diag(*BASIS_list)

        return np.asmatrix(BASIS)

    ## obtain basis functions at specific points
    #  @param self  The object pointer.
    #  @param x     points to evaluate [npoints x 1]
    #  @return      values of basis functions [npoints x nbasis]
    def _get_1d(self, x):
        # check input x
        x_vec = np.array(x)

        # shrink to range (edges have low capacity)
        range_min = self._range[0]
        range_max = self._range[1]
        x_vec = x_vec / (range_max - range_min) + range_min

        mpoints = len(x_vec)
        mbasis = self._nbasis

        BASIS = np.asmatrix(np.empty((mpoints, mbasis)))

        # first row is bias, always
        BASIS[:,0] = np.ones(shape=(mpoints,1))

        if (self._basisType == "rbf"):
            nrbf = (mbasis-1)
            BASIS[:,1:] = self._getBasisRbf(x, nbasis=nrbf)
        elif (self._basisType == "bernstein"):
            npoly = (mbasis-1)
            BASIS[:,1:] = self._getBasisBernstein(x, nbasis=npoly)
        else:
            raise NotImplementedError("type not available")

        # normalise weights (sum to 1)
        row_sums = BASIS.sum(axis=1)
        BASIS_norm = BASIS # / row_sums

        return BASIS_norm

    ## evaluates the rbf basis function
    #  @param self      The object pointer.
    #  @param x_vec     points to evaluate [npoints x 1]
    #  @param nbasis    number of basis functions
    #  @return          values of rbf basis functions [npoints x nbasis]
    def _getBasisRbf(self, x_vec, nbasis):
        x_vec = np.array(x_vec)

        mbasis = nbasis
        mpoints = len(x_vec)

        GAUS = np.empty(shape=(mpoints, mbasis))

        for (i, x_sca) in enumerate(x_vec):
            GAUS[i,:] = self._getBasisRbfVector(x_sca, mbasis)

        return np.asmatrix(GAUS)

    ## evaluates the rbf basis function for a single x
    #  @param self      The object pointer.
    #  @param x_sca     point to evaluate [1 x 1]
    #  @param nbasis    number of basis functions
    #  @return          values of rbf basis functions [1 x nbasis]
    def _getBasisRbfVector(self, x_sca, nbasis):
        gaus = np.empty(nbasis)

        gaus_loc_vec = np.linspace(0, 1, nbasis)
        # width according to S. Haykin. Neural Networks: A Comprehensive
        # Foundation (1994) pp. 236-284
        gaus_width = 1. / nbasis

        for i, gaus_loc in enumerate(gaus_loc_vec):
            gaus[i] = self._funcRbf(x_sca, gaus_loc, gaus_width)

        return np.asmatrix(gaus)

    ## evaluates a single rbf basis function for a single x
    #  @param self      The object pointer.
    #  @param x         point to evaluate [1 x 1]
    #  @param mu1       centre of rbf
    #  @param sig1      standard deviation of rbf
    #  @return          values of rbf basis functions [1 x nbasis]
    def _funcRbf(self, x, mu1, sig1):
        return np.exp(-(np.power((x-mu1), 2))/(2*sig1*sig1))

    ## evaluates the Bernstein basis function in [0, 1] using the formula \f$B(N,I)(X) = [N!/(I!*(N-I)!)] * (1-X)^(N-I) * X^I\f$
    #  @param self      The object pointer.
    #  @param x_vec     points to evaluate [npoints x 1]
    #  @param nbasis    number of basis functions (= polynomial degree + 1)
    #  @return          values of Bernstein basis functions [npoints x nbasis]
    def _getBasisBernstein(self, x_vec, nbasis):
        x_vec = np.array(x_vec)

        mbasis = nbasis
        mpoints = len(x_vec)

        BERN = np.empty(shape=(mpoints, mbasis))

        for (i, x_sca) in enumerate(x_vec):
            BERN[i,:] = self._getBasisBernsteinVector(x_sca, mbasis)

        return np.asmatrix(BERN)

    ## evaluates the Bernstein basis function in [0, 1] using the formula \f$B(N,I)(X) = [N!/(I!*(N-I)!)] * (1-X)^(N-I) * X^I\f$ for a single x
    #  @param self      The object pointer.
    #  @param x_sca     point to evaluate [1 x 1]
    #  @param nbasis    number of basis functions (= polynomial degree + 1)
    #  @return          values of Bernstein basis functions [npoints x nbasis]
    def _getBasisBernsteinVector(self, x_sca, nbasis):
        bern = np.empty(nbasis)
        n = nbasis - 1  # number of polynomials

        if (n == 0):
            bern[0] = 1
        else:
            bern[0] = 1 - x_sca  # bias
            bern[1] = x_sca  # linear

            for c in range(2, nbasis):
                bern[c] = x_sca * bern[c-1]

                for j in range(c-1, 0, -1):
                    bern[j] = x_sca * bern[j-1] + (1 - x_sca)*bern[j]

                bern[0] = (1 - x_sca)*bern[0]

        return np.asmatrix(bern)
