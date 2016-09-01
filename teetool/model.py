# models the trajectory data

from __future__ import print_function
import numpy as np
from numpy.linalg import det, inv, svd, pinv
import pathos.multiprocessing as mp
from pathos.helpers import cpu_count
import time, sys
import teetool as tt


class Model(object):
    """
    This class provides the interface to the probabilistic
    modelling of trajectories

    <description>
    """

    def __init__(self, cluster_data, settings):
        """
        cluster_data is a list of (x, Y)

        settings
        "model_type" = resampling, ML, or EM
        "ngaus": number of Gaussians to create for output
        REQUIRED for ML and EM
        "basis_type" = gaussian, bernstein
        "nbasis": number of basis functions
        """

        if "model_type" not in settings:
            raise ValueError("settings has no model_type")

        if type(settings["model_type"]) is not str:
            raise TypeError("expected string")

        if "ngaus" not in settings:
            raise ValueError("settings has no ngaus")

        if type(settings["ngaus"]) is not int:
            raise TypeError("expected int")

        if settings["model_type"] in ["ML", "EM"]:
            # required basis
            if "basis_type" not in settings:
                raise ValueError("settings has no basis_type")

            if "nbasis" not in settings:
                raise ValueError("settings has no nbasis")

            if settings["nbasis"] < 2:
                raise ValueError("nbasis should be larger than 2")

        # write global settings
        self._ndim = self._getDimension(cluster_data)

        # Fit x on a [0, 1] domain
        norm_cluster_data = self._normalise_data(cluster_data)

        # this part is specific for resampling
        if settings["model_type"] == "resampling":
            (mu_y, sig_y) = self._model_by_resampling(norm_cluster_data,
                                                      settings["ngaus"])
        elif settings["model_type"] == "ML":
            (mu_y, sig_y) = self._model_by_ml(norm_cluster_data,
                                              settings["ngaus"],
                                              settings["basis_type"],
                                              settings["nbasis"])
        elif settings["model_type"] == "EM":
            (mu_y, sig_y) = self._model_by_em(norm_cluster_data,
                                              settings["ngaus"],
                                              settings["basis_type"],
                                              settings["nbasis"])

        else:
            raise NotImplementedError("{0} not available".format(settings["model_type"]))

        # convert to cells
        (cc, cA) = self._getGMMCells(mu_y, sig_y, settings["ngaus"])

        # store values
        self._cc = cc
        self._cA = cA

    def eval(self, xx, yy, zz=None):
        """
        evaluates values in this grid [2d/3d] and returns values

        example grid:
        xx, yy, zz = np.mgrid[-60:60:20j, -10:240:20j, -60:60:20j]
        """

        # check values
        if not (xx.shape == yy.shape):
            raise ValueError("dimensions should equal (use np.mgrid)")

        nx = np.size(xx, 0)
        ny = np.size(yy, 1)
        if (self._ndim == 3):
            nz = np.size(zz, 2)

        # create two lists;
        # - index, idx
        # - position, pos
        list_idx = []
        list_pos = []

        if (self._ndim == 2):
            # 2d
            for ix in range(nx):
                for iy in range(ny):
                    x1 = xx[ix, 0]
                    y1 = yy[0, iy]

                    pos = np.mat([[x1], [y1]])

                    list_idx.append([ix, iy])
                    list_pos.append(pos)
        elif (self._ndim == 3):
            # 3d
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        x1 = xx[ix, 0, 0]
                        y1 = yy[0, iy, 0]
                        z1 = zz[0, 0, iz]

                        pos = np.mat([[x1], [y1], [z1]])

                        list_idx.append([ix, iy, iz])
                        list_pos.append(pos)
        else:
            raise NotImplementedError()

        # parallel processing
        ncores = cpu_count()
        p = mp.ProcessingPool(ncores)

        # output
        results = p.amap(self._gauss_logLc, list_pos)

        while not results.ready():
            # obtain intermediate results
            print(".", end="")
            sys.stdout.flush()
            time.sleep(3)

        print("") # new line

        # extract results
        list_val = results.get()

        # fill values here
        if (self._ndim == 2):
            # 2d
            s = np.zeros(shape=(nx, ny))

            for (i, idx) in enumerate(list_idx):
                # copy value in matrix
                s[idx[0], idx[1]] = list_val[i]
        elif (self._ndim == 3):
            # 3d
            s = np.zeros(shape=(nx, ny, nz))
            for (i, idx) in enumerate(list_idx):
                # copy value in matrix
                s[idx[0], idx[1], idx[2]] = list_val[i]
        else:
            return NotImplementedError()

        return s

    def _normalise_data(self, cluster_data):
        """
        normalises the x dimension
        """

        # determine minimum maximum
        tuple_min_max = self._getMinMax(cluster_data)

        for (i, (x, Y)) in enumerate(cluster_data):
            x = self._getNorm(x, tuple_min_max)  # normalise
            cluster_data[i] = (x, Y)  # overwrite

        return cluster_data

    def _model_by_resampling(self, cluster_data, ngaus):
        """
        returns (mu_y, sig_y) by resampling
        <description>
        """

        mdim = self._ndim

        # predict these values
        xp = np.linspace(0, 1, ngaus)

        yc = []  # list to put trajectories

        for (x, Y) in cluster_data:

            # array to fill
            yp = np.empty(shape=(ngaus, mdim))

            for d in range(mdim):
                yd = Y[:, d]
                yp[:, d] = np.interp(xp, x, yd)

            # single column
            yp1 = np.reshape(yp, (-1, 1), order='F')

            yc.append(yp1)

        # compute values

        mtraj = len(yc)  # number of trajectories

        # obtain average [mu]
        mu_y = np.zeros(shape=(mdim*ngaus, 1))

        for yn in yc:
            mu_y += yn

        mu_y = (mu_y / mtraj)

        # obtain standard deviation [sig]
        sig_y = np.zeros(shape=(mdim*ngaus, mdim*ngaus))

        for yn in yc:
            sig_y += ((yn - mu_y) * (yn - mu_y).transpose())

        sig_y = (sig_y / mtraj)

        return (mu_y, sig_y)

    def _model_by_ml(self, cluster_data, ngaus, type_basis, nbasis):
        """
        returns (mu_y, sig_y) by maximum likelihood (no noise assumed)

        <description>
        """

        ndim = self._ndim
        ntraj = len(cluster_data)

        # create a basis
        basis = self._Basis(type_basis, nbasis, ndim)

        wc = []

        for i, (xn, Y) in enumerate(cluster_data):
            yn = np.reshape(Y, newshape=(-1,1), order='F')

            Hn = basis.get(xn)

            wn = pinv(Hn) * yn

            wn = np.mat(wn)

            wc.append(wn)

        # obtain average [mu]
        mu_w = np.zeros(shape=(ndim*nbasis, 1))

        for wn in wc:
            mu_w += wn

        mu_w = np.mat(mu_w / ntraj)

        # obtain standard deviation [sig]
        sig_w = np.zeros(shape=(ndim*nbasis, ndim*nbasis))

        for wn in wc:
            sig_w += (wn - mu_w)*(wn - mu_w).transpose()

        sig_w = np.mat(sig_w / ntraj)

        # predict these values
        xp = np.linspace(0, 1, ngaus)
        Hp = basis.get(xp)

        mu_y = Hp * mu_w
        sig_y = Hp * sig_w * Hp.transpose()

        return (mu_y, sig_y)

    def _model_by_em(self, cluster_data, ngaus, type_basis, nbasis):
        """
        returns (mu_y, sig_y) by expectation-maximisation
        this allows noise to be modelled due to imperfect model or actual
        measurement noise

        <description>
        """

        return self._model_by_ml(cluster_data, ngaus, type_basis, nbasis)

    def _getMinMax(self, cluster_data):
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

    def _getNorm(self, x, tuple_min_max):
        """
        returns normalised array
        """
        (xmin, xmax) = tuple_min_max
        return ((x - xmin) / (xmax - xmin))

    def _getDimension(self, cluster_data):
        """
        returns dimension D of data
        """
        (_, Y) = cluster_data[0]
        (_, D) = Y.shape
        return D

    def _getGMMCells(self, mu_y, sig_y, ngaus):
        """
        return Gaussian Mixture Model (GMM) in cells
        """

        cc = []
        cA = []

        for m in range(ngaus):
            # single cell
            (c, A) = self._getMuSigma(mu_y, sig_y, m, ngaus)
            cc.append(c)
            cA.append(A)

        return (cc, cA)

    def _getMuSigma(self, mu_y, sig_y, npoint, ngaus):
        """
        returns (mu, sigma)
        """
        # mu_y [DM x 1]
        # sig_y [DM x DM]
        D = self._ndim

        # check range
        if ((npoint < 0) or (npoint >= ngaus)):
            raise ValueError("{0}, not in [0, {1}]".format(npoint, ngaus))

        c = np.empty(shape=(D, 1))
        A = np.empty(shape=(D, D))

        # select position
        for d_row in range(D):
            c[d_row, 0] = mu_y[(npoint+d_row*ngaus), 0]
            for d_col in range(D):
                A[d_row, d_col] = sig_y[(npoint+d_row*ngaus), (npoint+d_col*ngaus)]

        return (c, A)

    def _gauss(self, y, c, A):
        """
        returns value Gaussian
        """
        D = self._ndim

        p1 = 1 / np.sqrt(((2*np.pi)**D)*det(A))
        p2 = np.exp(-1/2*(y-c).transpose()*inv(A)*(y-c))

        return (p1*p2)

    def _gauss_logLc(self, y):
        """
        returns the log likelihood of a position based on model (in cells)
        """

        cc = self._cc
        cA = self._cA

        if (len(cc) != len(cA)):
            raise ValueError("expected size to match")

        M = len(cc)

        """
        should be zero, but this causes log infinity
        TODO: filter these results
        """

        # TODO remove hardcoding -- required for plotting
        py = 10**-30

        for m in range(M):
            c = cc[m]
            A = cA[m]
            py += self._gauss(y, c, A)  # addition of each Gaussian

        pyL = np.log(py) - np.log(M)  # division by number of Gaussians

        return pyL


    class _Basis(object):
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

            BASIS = tt.helpers.block_diag(*BASIS_list)

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
                val = self._funcRbf(x_sca, gaus_loc, gaus_width)
                #print([x_sca, gaus_loc, gaus_width, val])
                gaus[i] = val

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
