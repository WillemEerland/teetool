## @package teetool
#  This module contains the Model class
#
#  See Model class for more details

from __future__ import print_function
import numpy as np

from numpy.linalg import svd
from scipy.interpolate import griddata

import time, sys
import teetool as tt

import multiprocessing as mp
from functools import partial

## Model class provides the interface to the probabilistic modelling of trajectories
#
#  These trajectories are a single ensemble
class Model(object):

    ## Initialise a model
    # @param self object pointer
    # @param cluster_data list of (x, Y) trajectory data
    # @param settings a dictionary with
    # "model_type" = resampling, ML, or EM
    # "ngaus": number of Gaussians to create for output,
    # REQUIRED for ML and EM,
    # "basis_type" = gaussian, bernstein,
    # "nbasis": number of basis functions
    def __init__(self, cluster_data, settings):

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

        ## dimension of cluster_data
        self._ndim = tt.helpers.getDimension(cluster_data)



        # assume a gaussian stochastic process and model via one of these methods
        gp = tt.gaussianprocess.GaussianProcess(cluster_data,
                                                settings["ngaus"])

        # select method
        if settings["model_type"] == "resampling":
            (mu_y, sig_y, cc, cA) = gp.model_by_resampling()
        elif settings["model_type"].lower() == "ml":
            (mu_y, sig_y, cc, cA) = gp.model_by_ml(settings["basis_type"],
                                           settings["nbasis"])
        elif settings["model_type"].lower() == "em":
            (mu_y, sig_y, cc, cA) = gp.model_by_em(settings["basis_type"],
                                              settings["nbasis"])
        else:
            raise NotImplementedError("{0} not available".format(settings["model_type"]))

        # convert output back to normal values
        # (mu_y, sig_y) = self._norm2real(mu_y, sig_y, outline)

        # 'fix' to prevent the ellipsoids to collapse and cause numerical errors
        # sometimes referred to as a 'nudge'
        sig_y = sig_y + np.eye(sig_y.shape[0])*1e-12

        # convert to cells
        # (cc, cA) = self._getGMMCells(mu_y, sig_y, settings["ngaus"])

        ## mean vector [ngaus*ndim x 1]
        self._mu_y = mu_y
        ## covariance matrix [ngaus*ndim x ngaus*ndim]
        self._sig_y = sig_y
        ## mean vector [ndim x 1] in ngaus cells
        self._cc = cc
        ## covariance matrix [ndim x ndim] in ngaus cells
        self._cA = cA

        # create a list to store previous calculated values
        ## previously calculated values confidence region
        self._list_tube = []
        ## previously calculated values log-likelihood
        self._list_logp = []

    def _norm2real(self, mu_y, sig_y, outline):
        """returns the normalised values back to the original"""

        # TODO finish function

        return (mu_y, sig_y)


    def clear(self):
        """
        clears memory usage
        """
        self._list_tube = []
        self._list_logp = []

    def getMean(self):
        """
        returns the average trajectory [x, y, (z)]
        """

        ndim = self._ndim
        npoints = len(self._cc)

        mu_y = self._mu_y
        Y = np.reshape(mu_y, newshape=(-1, ndim), order='F')

        #Y = np.zeros(shape=(npoints, ndim))
        #for i, c in enumerate(self._cc):
        #    Y[i, :] = c

        if (ndim == 2):
            x = Y[:, 0]
            y = Y[:, 1]
            z = np.zeros_like(x) # fill with zeros
        elif (ndim == 3):
            x = Y[:, 0]
            y = Y[:, 1]
            z = Y[:, 2]

        Y_out = np.array([x, y, z]).T

        return Y_out


    def getSamples(self, nsamples):
        """
        return nsamples of the model
        """

        ndim = self._ndim

        mu_y = self._mu_y
        sig_y = self._sig_y

        npoints = np.size(mu_y, axis=0) / ndim

        mu_y = np.reshape(mu_y, newshape=(-1,))

        xp = np.linspace(0, 1, npoints)

        cluster_data = []

        np.random.seed(seed=10) # always same results

        for n in range(nsamples):

            yp = np.random.multivariate_normal(mu_y, sig_y, 1).T

            Yp = np.reshape(yp, newshape=(-1, ndim), order='F')

            cluster_data.append((xp, Yp))

        return cluster_data


    def _clusterdata2points(self, cluster_data):
        """returns the points Y [npoints x ndim]
        """

        Y_list = []

        for (x, Y) in cluster_data:
            Y_list.append(Y)

        return np.concatenate(Y_list, axis=0)

    def getKS(self, cluster_data, sigma_arr, nsamples=100):
        """calculated the ks value

        input:
            cluster_data    - data to evaluate against
            nsamples        - number of samples to use
            sigma_arr       - array which standard deviation to evalute
        """

        # only accept non-zero candidates
        sigma_arr = sigma_arr[sigma_arr > 0]

        if len(sigma_arr) == 0:
            raise ValueError("no positive values found")

        # in points
        Y = self._clusterdata2points(cluster_data)

        # obtain sampled data
        sampled_data = self.getSamples(nsamples=nsamples)

        # in points
        S = self._clusterdata2points(sampled_data)

        # test percentage inside
        lY = []
        lS = []
        for sigma in sigma_arr:
            # numbers true
            nY = self.isInside_pnts(Y, sdwidth=sigma, nellipse=30)
            # percentage
            pY = float(np.sum(nY)) / len(nY)
            # add
            lY.append(pY)
            # same
            nS = self.isInside_pnts(S, sdwidth=sigma, nellipse=30)
            pS = float(np.sum(nS)) / len(nS)
            lS.append(pS)

        lY = np.array(lY)
        lS = np.array(lS)

        ks = np.max(np.abs(lY - lS))

        return ks, lY, lS, sigma_arr

    def _getEllipse(self, c, A, sdwidth=1, nellipse=10):
        """
        evaluates the ellipse
        """

        ndim = self._ndim

        c = np.array(c)
        A = np.mat(A)

        # find the rotation matrix and radii of the axes
        [_, s, rotation] = svd(A)

        radii = sdwidth * np.sqrt(s)

        if ndim == 2:
            # 2d
            u = np.linspace(0.0, 2.0 * np.pi, nellipse)
            x = radii[0] * np.cos(u)
            y = radii[1] * np.sin(u)

            ellipse = np.empty(shape=(nellipse, ndim))
            ellipse[:,0] = x
            ellipse[:,1] = y
            ellipse = np.mat(ellipse)

            ap = ellipse * rotation.transpose() + c.transpose()

            # return np.mat(ellipse)

        if ndim == 3:
            # 3d
            # obtain sphere
            u = np.linspace(0.0, 2.0 * np.pi, nellipse)
            v = np.linspace(0.0, np.pi, nellipse)

            x = radii[0] * np.outer(np.cos(u), np.sin(v))
            y = radii[1] * np.outer(np.sin(u), np.sin(v))
            z = radii[2] * np.outer(np.ones_like(u), np.cos(v))

            x = x.reshape((-1,1), order='F')
            y = y.reshape((-1,1), order='F')
            z = z.reshape((-1,1), order='F')

            (nrows, ncols) = x.shape

            ap = np.zeros(shape=(nrows, ndim))

            ap[:, 0] = x.transpose()
            ap[:, 1] = y.transpose()
            ap[:, 2] = z.transpose()

            ap = tt.helpers.unique_rows(ap) # np.mat(ap).unique_rows()

            ap = ap * rotation.transpose() + c.transpose()

            # ap = np.concatenate([x, y, z], axis=1)

            # return np.mat(ap)

        ap = tt.helpers.unique_rows(ap)

        return np.mat(ap)


    def _getSample(self, c, A, nsamples=1, std=1):
        """
        returns nsamples samples of the given Gaussian
        """

        [U, S_diag, V] = svd(A)

        S = np.diag(S_diag)

        var_y = np.mat(np.real(U*np.sqrt(S)))

        ndim = self._ndim

        Y = np.zeros((nsamples, ndim))

        for i in range(nsamples):
            # obtain sample
            vecRandom = np.random.normal(size=(c.shape))
            Yi = c + var_y * (std * vecRandom)

            Y[i,:] = Yi.transpose()

        return np.mat(Y)

    def _getCoordsEllipse(self, nellipse=20, sdwidth=5):
        """
        returns an array of xy(z) coordinates

        nellipse is number of points in ellipsoid and sdwidth is the variance
        """

        ndim = self._ndim

        ngaus = len(self._cc)

        Y_list = []

        for i in range(ngaus):
            c = self._cc[i]
            A = self._cA[i]

            Yi = self._getEllipse(c, A, sdwidth, nellipse)

            Y_list.append(Yi)

        Y = np.concatenate(Y_list, axis=0)

        return np.mat(Y)

    def _eval_logp(self, Y_pos):
        """
        evaluates on a grid, aiming at the desired number of points
        """

        (npoints, _) = Y_pos.shape
        ndim = self._ndim

        # convert to list
        Y_pos_list = []

        for y in Y_pos:
            Y_pos_list.append(y)

        # create partial function (map only takes one argument)
        # ndim, cc, cA
        func = partial(tt.helpers.gauss_logLc, ndim=ndim, cc=self._cc, cA=self._cA)

        # parallel processing
        ncores = mp.cpu_count()
        p = mp.Pool(processes=ncores)

        # output - extract results
        list_val = p.map(func, Y_pos_list)

        # cleanup
        p.close()
        p.join()

        """
        # parallel processing
        p = mp.ProcessingPool()

        # output
        results = p.amap(self._gauss_logLc, Y_pos)

        while not results.ready():
            # obtain intermediate results
            print(".", end="")
            sys.stdout.flush()
            time.sleep(1)

        print("") # new line

        # extract results
        list_val = results.get()
        """

        s = np.array(list_val).squeeze()

        return s

    def _grid2points(self, xx, yy, zz=None):
        """
        returns an Y matrix for a given grid

        xx, yy, (zz) are mgrid

        """

        nx = np.size(xx, 0)
        ny = np.size(yy, 1)
        if (self._ndim == 3):
            nz = np.size(zz, 2)

        # create list;
        Y_pos = []
        Y_idx = []

        if (self._ndim == 2):
            # 2d
            for ix in range(nx):
                for iy in range(ny):
                    x1 = xx[ix, 0]
                    y1 = yy[0, iy]
                    pos = np.mat([x1, y1])
                    Y_pos.append(pos)
                    Y_idx.append([ix, iy])
        elif (self._ndim == 3):
            # 3d
            for ix in range(nx):
                for iy in range(ny):
                    for iz in range(nz):
                        x1 = xx[ix, 0, 0]
                        y1 = yy[0, iy, 0]
                        z1 = zz[0, 0, iz]
                        pos = np.mat([x1, y1, z1])
                        Y_pos.append(pos)
                        Y_idx.append([ix, iy, iz])
        else:
            raise NotImplementedError()

        Y_pos = np.concatenate(Y_pos, axis=0)
        Y_pos = np.array(Y_pos)

        #Y_idx = np.concatenate(Y_idx, axis=0)
        Y_idx = np.array(Y_idx)

        return (Y_pos, Y_idx)

    def _points2grid(self, s, Y_idx):
        """
        converts points to a matrix

        s is values np.array and Y_idx is position np.array
        """

        #print("s {0} {1}".format(np.min(s), np.max(s)))

        this_shape = np.max(Y_idx, axis=0)

        this_shape += 1  # one larger than indices

        ss = np.zeros(shape=this_shape, dtype=float)

        for i, y_idx in enumerate(Y_idx):
            # pass all positions (passes rows)

            if (self._ndim == 2):
                # 2d
                [ix, iy] = y_idx
                ss[ix, iy] = s[i]
            else:
                # 3d
                [ix, iy, iz] = y_idx
                ss[ix, iy, iz] = s[i]

        #print("ss {0} {1}".format(np.min(ss), np.max(ss)))

        return ss

    def isInside_grid(self, sdwidth, xx, yy, zz=None):
        """
        evaluate if points are inside a grid

        Input parameters:
            - sdwidth
            - xx
            - yy
            - zz (when 3d)
        """

        # check values
        if not (xx.shape == yy.shape):
            raise ValueError("dimensions should equal (use np.mgrid)")

        # ** check if this has been previously calculated

        ss = None
        # pass previous calculated versions
        for [ss1, sdwidth1, xx1, yy1, zz1] in self._list_tube:
            # check if exactly the same
            if (np.all(xx1==xx) and
                np.all(yy1==yy) and
                np.all(zz1==zz) and
                np.all(sdwidth1==sdwidth)):
                # copy
                ss = ss1

        if ss is None:
            # do the calculations

            # grid2points
            (Y_pos, Y_idx) = self._grid2points(xx, yy, zz)

            # evaluate points
            s = self.isInside_pnts(Y_pos, sdwidth)

            # points2grid
            ss = self._points2grid(s, Y_idx)

            # store results
            self._list_tube.append([ss, sdwidth, xx, yy, zz])


        # return values
        return ss

    def isInside_pnts(self, P, sdwidth=1, nellipse=50):
        """
        tests if points P NxD 'points' x 'dimensions' are inside the tube
        """

        ndim = self._ndim

        # obtain a list of points, representing the Gaussian and area between
        list_Y = self._get_point_cloud(sdwidth, nellipse)

        # P is an array
        P = np.array(P)

        # create partial function (map only takes one argument)
        func = partial(tt.helpers.in_hull, P)

        # parallel processing
        ncores = mp.cpu_count()
        p = mp.Pool(processes=ncores)

        # output - extract results
        list_these_inside = p.map(func, list_Y)

        # cleanup
        p.close()
        p.join()

        # convert to array
        arr_these_inside = np.array(list_these_inside).squeeze().transpose()

        # an array of bools (all FALSE, thus zeros)
        # FALSE = not inside
        # TRUE  = inside
        P_inside = np.any(arr_these_inside, axis=1)

        return P_inside


    def _get_point_cloud(self, sdwidth=1, nellipse=50):
        """
        returns a list with point clouds, representing the transition between Gaussians

        input paramters:
            - none
        """

        list_points_cloud = []

        ngaus = len(self._cc)

        # points of first Gaussian
        c = self._cc[0]
        A = self._cA[0]
        Yi = self._getEllipse(c, A, sdwidth, nellipse)

        for i in range(ngaus-1):

            # points of next Gaussian
            c = self._cc[i+1]
            A = self._cA[i+1]
            Yi1 = self._getEllipse(c, A, sdwidth, nellipse)

            # this is the 'cloud' to test
            Y = np.concatenate((Yi, Yi1), axis=0)

            # remove duplicates
            Y = tt.helpers.unique_rows(Y)

            list_points_cloud.append(Y)

            # new current Gaussian is next Gaussian
            Yi = Yi1.copy()

        return list_points_cloud


    def evalLogLikelihood(self, xx, yy, zz=None):
        """
        evaluates values in this grid [2d/3d] and returns values

        example grid:
        xx, yy, zz = np.mgrid[-60:60:20j, -10:240:20j, -60:60:20j]
        """

        # check values
        if not (xx.shape == yy.shape):
            raise ValueError("dimensions should equal (use np.mgrid)")

        ss = None
        # pass previous calculated versions
        for [ss1, xx1, yy1, zz1] in self._list_logp:
            # check if exactly the same
            if ( np.all(xx1==xx) and
                 np.all(yy1==yy) and
                 np.all(zz1==zz) ):
                # copy
                ss = ss1

        if ss is None:
            # do the calculations

            # grid2points
            (Y_pos, Y_idx) = self._grid2points(xx, yy, zz)

            # evaluate points
            s = self._eval_logp(Y_pos)

            # points2grid
            ss = self._points2grid(s, Y_idx)

            # replace NaN's with minimum
            ss[np.isnan(ss)] = np.nanmin(ss)

            # store values
            self._list_logp.append([ss, xx, yy, zz])

        return ss



    def getOutline(self, sdwidth=1):
        """
        returns the outline [xmin, xmax, ymin, ymax, zmin, zmax]

        input parameters:
            - sdwidth
        """

        # get maximum outline (borrow from other module)
        outline = tt.helpers.getMaxOutline(self._ndim)

        # by adding a bit, the bounds include the edges
        sdwidth += 0.1

        # obtain a list of points
        list_points_cloud = self._get_point_cloud(sdwidth, nellipse=20)

        for Y in list_points_cloud:

            for d in range(self._ndim):
                x = Y[:, d]
                xmin = x.min()
                xmax = x.max()
                if (outline[d*2] > xmin):
                    outline[d*2] = xmin
                if (outline[d*2+1] < xmax):
                    outline[d*2+1] = xmax

        return outline
