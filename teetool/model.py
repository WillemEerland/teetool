# models the trajectory data

# import support files here
import numpy as np
from numpy.linalg import det, inv

class Model(object):
    """
    This class provides the interface to the probabilistic modelling of trajectories

    <description>
    """

    def __init__(self, cluster_data, settings):
        """
        cluster_data is a list of (x, Y)

        settings for "model_type" = "resampling":
        "mpoints": number of points to resample
        """

        # (input checked in World)

        # extract settings
        M = settings["mpoints"]

        # write global settings
        self.D = self._getDimension(cluster_data)
        self.M = M

        # Fit x on a [0, 1] domain
        norm_cluster_data = self._normalise_data(cluster_data)

        """
        this part is specific for resampling
        """

        (mu_y, sig_y) = self._model_by_resampling(norm_cluster_data, M)

        """
        convert to cells
        """

        (cc, cA) = self._getGMMCells(mu_y, sig_y)

        """
        store values
        """

        self._cc = cc
        self._cA = cA

    def eval(self, x, y, z):
        """
        evaluates values in this grid [3D] and returns values

        example grid:
        x, y, z = np.ogrid[-60:60:20j, -10:240:20j, -60:60:20j]
        """

        nx = np.size(x, 0)
        ny = np.size(y, 1)
        nz = np.size(z, 2)

        ntotal = nx*ny*nz

        # fill values here
        s = np.zeros(shape=(nx,ny,nz))

        # TODO: parallel processing
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    x1 = x[ix,0,0]
                    y1 = y[0,iy,0]
                    z1 = z[0,0,iz]

                    xyz_loc = np.mat([[x1],[y1],[z1]])

                    s[ix,iy,iz] = temp = self._gauss_logLc(xyz_loc, self._cc, self._cA)

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

    def _model_by_resampling(self, cluster_data, M):
        """
        <description>
        """

        D = self.D

        # predict these values
        xp = np.linspace(0, 1, M)

        yc = [] # list to put trajectories

        for (x, Y) in cluster_data:

            # array to fill
            yp = np.empty(shape=(M, D))

            for d in range(D):
                yd = Y[:,d]
                yp[:,d] = np.interp(xp, x, yd)

            # single column
            yp1 = np.reshape(yp, (-1,1), order='F')

            yc.append(yp1)

        # compute values

        N = len(yc) # number of trajectories

        # obtain average [mu]
        mu_y = np.zeros(shape=(D*M,1))

        for yn in yc:
            mu_y += yn

        mu_y = (mu_y / N)

        # obtain standard deviation [sig]
        sig_y = np.zeros(shape=(D*M,D*M))

        for yn in yc:
            sig_y += ( (yn - mu_y) * (yn - mu_y).transpose() )

        sig_y = (sig_y / N)

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

    def _getGMMCells(self, mu_y, sig_y):
        """
        return Gaussian Mixture Model (GMM) in cells
        """

        M = self.M

        cc = []
        cA = []

        for m in range(M):
            # single cell
            (c, A) = self._getMuSigma(mu_y, sig_y, m)
            cc.append(c)
            cA.append(A)

        return (cc, cA)

    def _getMuSigma(self, mu_y, sig_y, npoint):
        """
        returns (mu, sigma)
        """
        # mu_y [DM x 1]
        # sig_y [DM x DM]
        D = self.D
        M = self.M

        # check range
        if ( (npoint < 0) or (npoint >= M) ):
            raise ValueError("{0}, not in [0, {1}]".format(npoint, M))

        c = np.empty(shape=(D,1))
        A = np.empty(shape=(D,D))

        # select position
        for d_row in range(D):
            c[d_row,0] = mu_y[(npoint+d_row*M),0]
            for d_col in range(D):
                A[d_row,d_col] = sig_y[(npoint+d_row*M), (npoint+d_col*M)]

        return (c, A)

    def _gauss(self, y, c, A):
        """
        returns value Gaussian
        """
        D = self.D

        p1 = 1 / np.sqrt( ((2*np.pi)**D)*det(A)  )
        p2 = np.exp( -1/2*(y-c).transpose()*inv(A)*(y-c) )

        return (p1*p2)

    def _gauss_logLc(self, y, cc, cA):
        """
        returns the log likelihood of a position based on model (in cells)
        """

        if ( len(cc) != len(cA) ):
			raise ValueError("expected size to match")

        M = len(cc)

        py = 0

        for m in range(M):
            c = cc[m]
            A = cA[m]
            py += self._gauss(y, c, A) # addition of each Gaussian

        if (py == 0):
            pyL = np.nan
        else:
            pyL = np.log(py) - np.log(M) # division by number of Gaussians

        return pyL
