# does things with basis functions

import numpy as np
from scipy.linalg import block_diag


class Basis(object):
    """This class provides the interface between models

    <description>
    """

    def __init__(self, basisType, nbasis, ndim):
        """
        initialises a basis type

        - 'rbf' uniformly distributed radial basis functions
        - 'bernstein' Bernstein polynomials
        """

        SUPPORTED_FUNCTIONS = ["rbf", "bernstein"]

        if basisType not in SUPPORTED_FUNCTIONS:
            raise NotImplementedError("{0} type not supported, only {1}".format(basisType, SUPPORTED_FUNCTIONS))

        self._basisType = basisType
        self._nbasis = nbasis  # number of basis functions
        self._ndim = ndim  # number of dimensions
        margin = 1/(2*nbasis)
        self._range = [0+margin, 1-margin]

    def get(self, x):
        """
        return values of basis
        """

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

        return np.mat(BASIS)


    def _get_1d(self, x):
        """
        returns values of basis, single dimension
        """

        # check input x
        x_vec = np.array(x)

        # shrink to range (edges have low capacity)
        range_min = self._range[0]
        range_max = self._range[1]
        x_vec = x_vec / (range_max - range_min) + range_min

        mpoints = len(x_vec)
        mbasis = self._nbasis

        BASIS = np.mat(np.empty((mpoints, mbasis)))

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


    def _getBasisRbf(self, x_vec, nbasis):
        """
        returns a matrix, with K columns, and size(x) rows
        x: input vector
        K: number of basis functions
        """

        x_vec = np.array(x_vec)

        mbasis = nbasis
        mpoints = len(x_vec)

        GAUS = np.empty(shape=(mpoints, mbasis))

        for (i, x_sca) in enumerate(x_vec):
            GAUS[i,:] = self._getBasisRbfVector(x_sca, mbasis)

        return np.mat(GAUS)

    def _getBasisRbfVector(self, x_sca, nbasis):
        """
        gaussian -- vector

        x_sca : scalar value
        nbasis : number of rbf's
        """

        gaus = np.empty(nbasis)

        gaus_loc_vec = np.linspace(0, 1, nbasis)
        # width according to S. Haykin. Neural Networks: A Comprehensive
        # Foundation (1994) pp. 236-284
        gaus_width = 1. / nbasis

        for i, gaus_loc in enumerate(gaus_loc_vec):
            gaus[i] = self._funcRbf(x_sca, gaus_loc, gaus_width)

        return np.mat(gaus)

    def _funcRbf(self, x, mu1, sig1):
        """
        radial basis -- function
        """

        return np.exp(-(np.power((x-mu1), 2))/(2*sig1*sig1))

    def _getBasisBernstein(self, x_vec, nbasis):
        """
        evaluates the Bernstein polynomials in [0, 1]
        The formula it uses is:
        B(N,I)(X) = [N!/(I!*(N-I)!)] * (1-X)^(N-I) * X^I
        ---
        returns a matrix, with n + 1 columns, and size(x) rows
        x_vec: input vector
        K: number of basis functions
        """

        x_vec = np.array(x_vec)

        mbasis = nbasis
        mpoints = len(x_vec)

        BERN = np.empty(shape=(mpoints, mbasis))

        for (i, x_sca) in enumerate(x_vec):
            BERN[i,:] = self._getBasisBernsteinVector(x_sca, mbasis)

        return np.mat(BERN)

    def _getBasisBernsteinVector(self, x_sca, nbasis):
        """
        bernstein -- vector
        """

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

        return np.mat(bern)
