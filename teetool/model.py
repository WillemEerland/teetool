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
        self._mu_y = mu_y
        self._sig_y = sig_y
        #
        self._cc = cc
        self._cA = cA

    def getSamples(self, nsamples):
        """
        return nsamples
        """

        ndim = self._ndim

        mu_y = self._mu_y
        sig_y = self._sig_y

        npoints = np.size(mu_y, axis=0) / ndim

        [U, S_diag, V] = svd(sig_y)

        S = np.diag(S_diag)

        var_y = np.mat(np.real(U*np.sqrt(S)))

        xp = np.linspace(0, 1, npoints)

        cluster_data = []

        np.random.seed(seed=10) # always same results

        for n in range(nsamples):
            vecRandom = np.random.normal(size=(mu_y.shape))
            yp = mu_y + var_y * vecRandom
            Yp = np.reshape(yp, (-1, ndim), order='F')
            cluster_data.append((xp, Yp))

        return cluster_data


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

        for (xn, Yn) in cluster_data:

            # array to fill
            yp = np.empty(shape=(ngaus, mdim))

            for d in range(mdim):
                ynd = Yn[:, d]
                yp[:, d] = np.interp(xp, xn, ynd)

            # single column
            yp1 = np.reshape(yp, (-1, 1), order='F')

            yc.append(yp1)

        # compute values

        ntraj = len(yc)  # number of trajectories

        # obtain average [mu]
        mu_y = np.zeros(shape=(mdim*ngaus, 1))

        for yn in yc:
            mu_y += yn

        mu_y = (mu_y / ntraj)

        # obtain standard deviation [sig]
        sig_y_sum = np.zeros(shape=(mdim*ngaus, mdim*ngaus))

        for yn in yc:
            sig_y_sum += (yn - mu_y) * (yn - mu_y).transpose()

        sig_y = np.mat(sig_y_sum / ntraj)

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
        sig_w_sum = np.zeros(shape=(ndim*nbasis, ndim*nbasis))

        for wn in wc:
            sig_w_sum += (wn - mu_w)*(wn - mu_w).transpose()

        sig_w = np.mat(sig_w_sum / ntraj)

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

        ndim = self._ndim
        ntraj = len(cluster_data)

        Mstar = 0
        for (xn, Yn) in cluster_data:
            Mstar += np.size(xn)

        # create a basis
        basis = self._Basis(type_basis, nbasis, ndim)

        # prepare data
        yc = []
        Hc = []

        for (xn, Yn)  in cluster_data:
            # data
            yn = np.reshape(Yn, newshape=(-1,1), order='F')
            Hn = basis.get(xn)
            # add to list
            yc.append(yn)
            Hc.append(Hn)

        # hardcoded parameters
        MAX_ITERATIONS = 2001  # maximum number of iterations
        CONV_LIKELIHOOD = 1e-3  # stop convergence
        # min_eig = 10**-6  # minimum eigenvalue (numerical trick)
        BETA_EM_LIMIT = 1e8  # maximum accuracy

        # initial variables
        BETA_EM = 1000.
        mu_w = np.zeros(shape=(nbasis*ndim, 1))
        sig_w = np.mat(np.eye(nbasis*ndim))
        sig_w_inv = inv(sig_w)

        loglikelihood_previous = np.inf

        for i_iter in range(MAX_ITERATIONS):

            Ewc = []
            Ewwc = []

            # Expectation (54) (55)
            for n  in range(ntraj):
                # data
                yn = yc[n]
                Hn = Hc[n]

                # calculate S :: (50)
                Sn_inv = sig_w_inv + np.multiply(BETA_EM,(Hn.transpose()*Hn))
                Sn = np.mat(inv(Sn_inv))

                Ewn = (Sn *((np.multiply(BETA_EM,(Hn.transpose()*yn))) + ((sig_w_inv*mu_w))))

                Ewn = np.mat(Ewn)

                # BISHOP (2.62)
                Ewnwn = Sn + Ewn*Ewn.transpose()

                Ewnwn = np.mat(Ewnwn)

                # store
                Ewc.append(Ewn);
                Ewwc.append(Ewnwn);

            #  Maximization :: (56), (57)

            # E [ MU ]
            mu_w_sum = np.zeros(shape=(nbasis*ndim, 1));

            for n  in range(ntraj):
                # extract data
                Ewn = Ewc[n]
                # sum
                mu_w_sum += Ewn

            mu_w = np.mat(mu_w_sum / ntraj)

            # E [ SIGMA ]
            sig_w_sum = np.zeros((nbasis*ndim, nbasis*ndim));

            for n  in range(ntraj):
                # extract data
                yn = yc[n]
                Hn = Hc[n]
                Ewn = Ewc[n]
                Ewnwn = Ewwc[n]

                # sum
                SIGMA_n = Ewnwn - 2.*(mu_w*Ewn.transpose()) + mu_w*mu_w.transpose()
                sig_w_sum += SIGMA_n

            sig_w = np.mat(sig_w_sum / ntraj)

            sig_w_inv = inv(sig_w)

            """
            # !! nearest SPD inverse
            [U, S_diag, V] = svd(sig_w)


            ln_det_Sigma = np.log(np.prod(S_diag))

            # (optional) check eigenvalues
            while np.isinf(ln_det_Sigma):
                # adjust value
                S_diag[S_diag < min_eig] = min_eig

                # check
                ln_det_Sigma = np.log( np.prod(S_diag) )

                if np.isinf(ln_det_Sigma):
                    # if still not good, add a little
                    min_eig = min_eig + 10**-6

            sig_w = np.mat(U * np.diag( S_diag ) * V)
            """

            # E [BETA]
            BETA_sum_inv = 0.;

            for n  in range(ntraj):
                # extract data
                yn = yc[n]
                Hn = Hc[n]
                Ewn = Ewc[n]
                Ewnwn = Ewwc[n]

                BETA_sum_inv += np.dot(yn.transpose(),yn) - 2.*(np.dot(yn.transpose(),(Hn*Ewn))) + np.trace((Hn.transpose()*Hn)*Ewnwn)

            BETA_EM = np.mat((ndim*Mstar) / BETA_sum_inv)

            # limit BETA_EM (how much accuracy is relevant?)
            #if BETA_EM > BETA_EM_LIMIT:
            # BETA_EM = BETA_EM_LIMIT

            # ////  log likelihood ///////////

            # // ln( p(Y|w) - likelihood
            loglikelihood_pYw_sum = 0.;

            for n  in range(ntraj):
                # extract data
                yn = yc[n]
                Hn = Hc[n]
                Ewn = Ewc[n]
                Ewnwn = Ewwc[n]

                # loglikelihood_pYw_sum = loglikelihood_pYw_sum + ((yn.')*yn - 2*(yn.')*(Hn*Ewn) + trace(((Hn.')*Hn)*Ewnwn));
                loglikelihood_pYw_sum += np.dot(yn.transpose(),yn) - 2.*(np.dot(yn.transpose(),(Hn*Ewn))) + np.trace((Hn.transpose()*Hn)*Ewnwn)

            #  loglikelihood_pYw =  + ((Mstar*D) / 2) * log(2*pi) - ((Mstar*D) / 2) * log( BETA_EM ) + (BETA_EM/2) * loglikelihood_pYw_sum;
            loglikelihood_pYw = (Mstar*ndim / 2.) * np.log(2.*np.pi) - (Mstar*ndim / 2.) * np.log(BETA_EM) + (BETA_EM / 2.) * loglikelihood_pYw_sum

            # // ln( p(w) ) - prior
            loglikelihood_pw_sum = 0.;

            for n  in range(ntraj):
                # extract data
                Ewn = Ewc[n]
                Ewnwn = Ewwc[n]

                # loglikelihood_pw_sum = loglikelihood_pw_sum + trace( (LAMBDA_EM)*( Ewnwn - 2*MU_EM*(Ewn.') + (MU_EM*(MU_EM.')) ) );
                loglikelihood_pw_sum += np.trace(sig_w_inv*(Ewnwn - 2.*mu_w*Ewn.transpose() + mu_w*mu_w.transpose()))

            # loglikelihood_pw = + ((N*J*D) / 2) * log(2*pi) + (N/2) * ln_det_Sigma + (1/2) * loglikelihood_pw_sum;
            loglikelihood_pw = (ntraj*nbasis*ndim/2.)*np.log(2*np.pi) + (ntraj/2.)*np.log(det(sig_w)) + (1./2.)*loglikelihood_pw_sum

            loglikelihood_pY = loglikelihood_pYw + loglikelihood_pw

            # // check convergence
            loglikelihood_diff = np.abs(loglikelihood_pY - loglikelihood_previous)

            if np.isfinite(loglikelihood_pY):
                # check
                if (loglikelihood_diff < CONV_LIKELIHOOD):
                    break
            else:
                # not a valid loglikelihood
                print("warning: not a finite loglikelihood")
                break

            # output
            #if (i_iter % 100 == 0):
            #    print("{0} {1} {2}".format(i_iter, loglikelihood_pY, min_eig))

            # store previous log_likelihood
            loglikelihood_previous = loglikelihood_pY

        # predict these values
        xp = np.linspace(0, 1, ngaus)
        Hp = basis.get(xp)

        mu_y = Hp * mu_w
        sig_y = Hp * sig_w * Hp.transpose()

        return (mu_y, sig_y)

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

        M = len(cc)

        """
        should be zero, but this causes log infinity
        TODO: filter these results
        """

        # TODO remove hardcoding -- required for plotting
        py = 10**-40

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
