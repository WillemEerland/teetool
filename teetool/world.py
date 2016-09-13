#  "main" source file for trajectory analysis in this package.
# Clusters of trajectories are added here, modelled accordingly
# (whatever settings desired), and visualised in whatever is
# desired (single / multiple clusters)

import numpy as np
import teetool as tt


class World(object):
    """
    This class provides the interface for the trajectory analysis tool.

    <description>

    Initialisation arguments:
     - name
     - dimension

    Properties:
     -
     -
    """

    def __init__(self, name="", ndim=3):
        """
        name: name of World
        ndim: dimension of world (2d or 3d)
        <description>
        """

        # validate name
        if type(name) is not str:
            raise TypeError("expected string, not {0}".format(type(name)))

        # validate dimension
        if type(ndim) is not int:
            raise TypeError(
                "expected integer, not {0}".format(type(ndim)))

        if (ndim != 2) and (ndim != 3):
            raise ValueError(
                "expected dimensionality 2 or 3, not {0}".format(ndim))

        # set values
        self._name = name
        self._D = ndim

        # initial parameters
        self._clusters = []  # list holding clusters
        # these parameters define the grid
        self._real_outline = self._getDefaultOutline(ndim)
        self._resolution = self._getDefaultResolution(ndim)

        # default value
        self.fraction_to_expand = 0.1

    def overview(self):
        """
        prints overview in console
        """

        print("*** overview [{0}] ***".format(self._name))

        for (i, this_cluster) in enumerate(self._clusters):

            if ("model" in this_cluster):
                has_model = "*"
            else:
                has_model = "-"

            if ("logp" in this_cluster):
                has_logp = "*"
            else:
                has_logp = "-"

            print("{0} [{1}] [{2}] [{3}]".format(
                        i, this_cluster["name"], has_model, has_logp))

    def addCluster(self, cluster_data, cluster_name=""):
        """
        <description>

        Input arguments:
            - aCluster: list with tuples (x, Y) representing trajectory data
            - name: a string with the name of the cluster
        """

        # validate cluster_name
        if type(cluster_name) is not str:
            raise TypeError(
                "expected string, not {0}".format(type(cluster_name)))

        # validate cluster_data
        if type(cluster_data) is not list:
            raise TypeError(
                "expected list, not {0}".format(type(cluster_data)))

        # validate trajectory_data
        for (i, trajectory_data) in enumerate(cluster_data):
            # check type
            if type(trajectory_data) is not tuple:
                raise TypeError(
                    "expected tuple, item {0} is a {1}".format(
                        i, type(trajectory_data)))
            # check values x [M x 1], Y [M x D]
            (x, Y) = trajectory_data
            (M, D) = Y.shape
            if (D != self._D):
                raise ValueError("dimension not correct")
            if (M != np.size(x, 0)):
                raise ValueError("number of data-points do not match")
            # check if all finite
            if not np.isfinite(x).all():
                raise ValueError("x holds non-finite values")

        # add new cluster [ holds "name" and "data" ]
        new_cluster = {}

        new_cluster["name"] = cluster_name
        new_cluster["data"] = cluster_data

        self._clusters.append(new_cluster)

        self._check_real_outline(cluster_data)  # extend outline, if required

    def getName(self):
        """
        returns name, if any, otherwise returns None
        """

        if self._name == "":
            return None
        else:
            return self._name



    def getCluster(self, icluster):
        """
        returns a single cluster
        """

        # TODO checks, and optional to provide name

        return self._clusters[icluster]

    def getClusters(self):
        """
        returns the clusters
        """

        return self._clusters

    def _check_icluster(self, icluster):
        """
        check validity int icluster input
        """

        if type(icluster) is not int:
            raise TypeError("expected integer, not {0}".format(type(icluster)))

        nclusters = len(self._clusters)
        if ((icluster < 0) or (icluster >= nclusters)):
            raise ValueError(
                "{0} not in range [0,{1}]".format(icluster, nclusters))

    def getSamples(self, icluster, nsamples=50):
        """
        returns samples (x, Y) list
        """

        # check validity
        self._check_icluster(icluster)

        # extract
        this_cluster = self._clusters[icluster]

        generated_samples = this_cluster["model"].getSamples(nsamples)

        return generated_samples

    def buildModel(self, icluster, settings):
        """
        creates a model with these settings
        model_type: [resample]
        mgaus: number of Gaussians (e.g. 50-100)
        """

        # check validity
        self._check_icluster(icluster)

        # extract
        this_cluster = self._clusters[icluster]

        # build a new model
        new_model = tt.model.Model(this_cluster["data"], settings)

        # overwrite
        this_cluster["model"] = new_model
        self._clusters[icluster] = this_cluster

    def buildLogProbality(self, icluster):
        """
        builds a log-probability grid
        """

        if type(icluster) is not int:
            raise TypeError("expected integer, not {0}".format(type(icluster)))

        nclusters = len(self._clusters)
        if ((icluster < 0) or (icluster >= nclusters)):
            raise ValueError(
                "{0} not in range [0,{1}]".format(icluster, nclusters))

        # extract
        this_cluster = self._clusters[icluster]


        if (self._D == 2):
            # 2d
            [xx, yy] = self.getGrid(ndim=2)
            temp = this_cluster["model"].eval(xx, yy)

        if (self._D == 3):
            # 3d
            [xx, yy, zz] = self.getGrid(ndim=3)
            temp = this_cluster["model"].eval(xx, yy, zz)


        #(Y, s) = this_cluster["model"]._eval_random()

        this_cluster["logp"] = temp

        # overwrite
        self._clusters[icluster] = this_cluster

    def _getDefaultOutline(self, ndim):
        """
        returns default outline based on dimensionality
        """
        defaultOutline = []

        for d in range(ndim):
            defaultOutline.append(np.inf)  # min
            defaultOutline.append(-np.inf)  # max

        return defaultOutline

    def _getDefaultResolution(self, ndim):
        """
        returns default outline based on dimensionality
        """
        defaultResolution = []

        for d in range(ndim):
            defaultResolution.append(20)  # default resolution

        return defaultResolution

    def setResolution(self, xstep, ystep, zstep=None):
        """
        sets the resolution
        WARNING: removes any existing loglikelihood calculations
        """

        # remove existing likelihood calculations
        for i in range(len(self._clusters)):
            self._clusters[i].pop("logp", None)

        # new resolution
        if (self._D) == 2:
            self._resolution = [xstep, ystep]

        if (self._D == 3):
            self._resolution = [xstep, ystep, zstep]

    def getGrid(self, ndim, resolution=None):
        """
        returns the grid used calculate the log-likelihood on
        """

        if resolution is None:
            resolution = self._resolution

        #outline = self.getRealOutline()
        outline = self.getExpandedOutline()

        #print(outline1)
        #print(outline)

        if (ndim == 2):
            # 2d
            [xmin, xmax, ymin, ymax] = outline[0:4]
            [xstep, ystep] = resolution[0:2]
            # 2d
            res = np.mgrid[xmin:xmax:np.complex(0, xstep),
                           ymin:ymax:np.complex(0, ystep)]

        if (ndim == 3):
            # 3d
            [xmin, xmax, ymin, ymax, zmin, zmax] = outline[0:6]
            [xstep, ystep, zstep] = resolution[0:3]
            # 3d
            res = np.mgrid[xmin:xmax:np.complex(0, xstep),
                           ymin:ymax:np.complex(0, ystep),
                           zmin:zmax:np.complex(0, zstep)]

        return res

    def getRealOutline(self):
        """
        returns real outline of data
        """
        return self._real_outline

    def getExpandedOutline(self):
        """
        returns expanded outline of data (via fraction_to_expand)
        """

        real_outline = self.getRealOutline()

        ndim = self._D

        # init
        if (self._D == 2):
            expanded_outline = [0, 0, 0, 0]
        if (self._D == 3):
            expanded_outline = [0, 0, 0, 0, 0, 0]

        for d in range(ndim):
            pos1 = d*2
            pos2 = d*2 + 2
            [xmin, xmax] = real_outline[pos1:pos2]
            xdif = xmax - xmin

            expanded_outline[pos1] = real_outline[pos1] - self.fraction_to_expand*xdif
            expanded_outline[pos1+1] = real_outline[pos1+1] + self.fraction_to_expand*xdif

        if (self._D == 3):
            expanded_outline[4] = 0 # set minimum altitude to zero

        # convert to numpy array
        expanded_outline = np.array(expanded_outline)

        return expanded_outline

    def _check_real_outline(self, cluster_data):
        """
        calculate maximum outline from list and store
        """

        for (x, Y) in cluster_data:

            for d in range(self._D):
                x = Y[:, d]
                xmin = x.min()
                xmax = x.max()
                if (self._real_outline[d*2] > xmin):
                    self._real_outline[d*2] = xmin
                if (self._real_outline[d*2+1] < xmax):
                    self._real_outline[d*2+1] = xmax
