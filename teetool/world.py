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

    def __init__(self, name="", ndim=3, resolution=[10, 10, 10]):
        """
        initialises a World

        input parameters:
            - name: name of World
            - ndim: dimension of world (2d or 3d)
            - nres: sets the resolution of the grid
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
        self._ndim = ndim

        # initial parameters
        self._clusters = []  # list holding clusters

        # these parameters define the grid
        self._resolution = resolution

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

            print("{0} [{1}] [{2}]".format(
                        i, this_cluster["name"], has_model))

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
            if (D != self._ndim):
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
        # obtain the outline of this data
        new_cluster["outl"] = self._get_outline_cluster(cluster_data)

        # add cluster to the list
        self._clusters.append(new_cluster)

    def getName(self):
        """
        returns name, if any, otherwise returns None
        """

        if self._name == "":
            return None
        else:
            return self._name

    def getCluster(self, list_icluster=None):
        """
        returns a single cluster
        """

        list_icluster = self._check_list_icluster(list_icluster)

        # return clusters in list
        clusters = []
        for i in list_icluster:
            clusters.append(self._clusters[i])
        return clusters

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

    def _check_list_icluster(self, list_icluster):
        """
        check validity of list of integers
        """

        # default
        if (list_icluster == None):
            # all
            list_icluster = range(len(self._clusters))

        if type(list_icluster) is not list:
            raise TypeError("expected list, not {0}".format(type(list_icluster)))

        for icluster in list_icluster:
            # check validity
            self._check_icluster(icluster)

        return list_icluster

    def isInside(self, P, sdwidth=1, list_icluster=None):
        """
        returns list of bools, whether or not the points P are inside any of the models

        list_icluster can be set to limit the check to a single values
        """

        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        list_inside = []

        return list_inside


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

    def buildModel(self, settings, list_icluster=None):
        """
        creates a model

        settings are
        model_type: [resample]
        mgaus: number of Gaussians (e.g. 50-100)
        """

        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        for icluster in list_icluster:
            # extract
            this_cluster = self._clusters[icluster]

            # build a new model
            new_model = tt.model.Model(this_cluster["data"], settings)

            # overwrite
            this_cluster["model"] = new_model
            self._clusters[icluster] = this_cluster

    def getMean(self, list_icluster=None):
        """
        returns the mean trajectory [x, y, z] for list_icluster
        """

        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        Y_list = []

        for icluster in list_icluster:
            # extract
            this_cluster = self._clusters[icluster]
            # obtain mean
            Y = this_cluster["model"].getMean()
            # append to list
            Y_list.append(Y)

        return Y_list


    def getTube(self, list_icluster=None, sdwidth=1, resolution=None, z=None):
        """
        return (ss_list, [xx, yy, zz]) of models that fall within sdwidth

        Input parameters:
            - list_icluster
            - sdwidth
            - z
        """

        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        # obtain the outline for the grid
        outline = self._get_outline_tube(sdwidth, list_icluster)

        # if z is set, reduce outline to four parameters
        if z is not None:
            outline = outline[:4]

        # obtain grid to evaluate on
        [xx, yy, zz] = self._getGrid(outline, resolution)

        # temporary adjust arrays
        if z is not None:
            xx = np.reshape(xx, newshape=(xx.shape[0], xx.shape[1], 1))
            yy = np.reshape(yy, newshape=(yy.shape[0], yy.shape[1], 1))
            zz = np.ones_like(xx)*1.0*z;

        # values returned
        ss_list = []

        for icluster in list_icluster:
            # extract
            this_cluster = self._clusters[icluster]

            ss = this_cluster["model"].isInside_grid(sdwidth, xx, yy, zz)

            if z is not None:
                ss = np.reshape(ss, newshape=(xx.shape[0], xx.shape[1]))

            ss_list.append(ss)

        # re-adjust
        if z is not None:
            xx = np.reshape(xx, newshape=(xx.shape[0], xx.shape[1]))
            yy = np.reshape(yy, newshape=(yy.shape[0], yy.shape[1]))
            zz = None

        return (ss_list, [xx, yy, zz])

    def getLogLikelihood(self, list_icluster=None, resolution=None, z=None):
        """
        return (ss_list, [xx, yy, zz]) of models and corresponding log-likelihood

        Input parameters:
            - list_icluster
        """

        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        # obtain outline
        outline = self._get_outline_expanded(list_icluster)

        # if z is set, reduce outline to four parameters
        if z is not None:
            outline = outline[:4]

        # obtain grid to evaluate on
        [xx, yy, zz] = self._getGrid(outline, resolution)

        # temporary adjust arrays
        if z is not None:
            xx = np.reshape(xx, newshape=(xx.shape[0], xx.shape[1], 1))
            yy = np.reshape(yy, newshape=(yy.shape[0], yy.shape[1], 1))
            zz = np.ones_like(xx)*1.0*z;

        # values returned
        ss_list = []

        for icluster in list_icluster:
            # extract
            this_cluster = self._clusters[icluster]

            ss = this_cluster["model"].evalLogLikelihood(xx, yy, zz)

            if z is not None:
                ss = np.reshape(ss, newshape=(xx.shape[0], xx.shape[1]))

            ss_list.append(ss)

        # re-adjust
        if z is not None:
            xx = np.reshape(xx, newshape=(xx.shape[0], xx.shape[1]))
            yy = np.reshape(yy, newshape=(yy.shape[0], yy.shape[1]))
            zz = None

        return (ss_list, [xx, yy, zz])

    def _getGrid(self, outline, resolution=None):
        """
        returns the grid

        based on outline and resolution
        """

        # default resolution
        if resolution is None:
            resolution = self._resolution

        # use expanded grid for calculations
        #outline = self._get_outline_expanded(list_icluster)

        [xx, yy, zz] = tt.helpers.getGridFromResolution(outline, resolution)

        return [xx, yy, zz]


    def _get_outline_expanded(self, list_icluster=None, fraction_to_expand=0.1):
        """
        returns expanded outline of data (via fraction_to_expand)
        """

        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        original_outline = self._get_outline(list_icluster)

        ndim = self._ndim

        # init
        if (ndim == 2):
            expanded_outline = [0, 0, 0, 0]
        if (ndim == 3):
            expanded_outline = [0, 0, 0, 0, 0, 0]

        for d in range(ndim):
            pos1 = d*2
            pos2 = d*2 + 2
            [xmin, xmax] = original_outline[pos1:pos2]
            xdif = xmax - xmin

            expanded_outline[pos1] = original_outline[pos1] - fraction_to_expand*xdif
            expanded_outline[pos1+1] = original_outline[pos1+1] + fraction_to_expand*xdif

        # convert to numpy array
        expanded_outline = np.array(expanded_outline)

        return expanded_outline

    def _get_outline(self, list_icluster=None):
        """
        returns the outline of specified clusters

        list_icluster is list of clusters, if None, show all
        """

        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        global_outline = tt.helpers.getMaxOutline(self._ndim)

        for icluster in list_icluster:

            # outline
            cluster = self._clusters[icluster]

            local_outline = cluster["outl"]

            for d in range(self._ndim):

                # cluster specific
                xmin = local_outline[d*2]
                xmax = local_outline[d*2+1]

                if xmin < global_outline[d*2]:
                    global_outline[d*2] = xmin

                if xmax > global_outline[d*2+1]:
                    global_outline[d*2+1] = xmax

        return global_outline

    def _get_outline_tube(self, sdwidth=1, list_icluster=None):
        """
        returns the outline of specified clusters for the tube

        list_icluster is list of clusters, if None, show all
        """

        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        global_outline = tt.helpers.getMaxOutline(self._ndim)

        for icluster in list_icluster:
            # pass clusters
            this_cluster = self._clusters[icluster]

            local_outline = this_cluster["model"].getOutline(sdwidth)

            for d in range(self._ndim):

                # cluster specific
                xmin = local_outline[d*2]
                xmax = local_outline[d*2+1]

                if xmin < global_outline[d*2]:
                    global_outline[d*2] = xmin

                if xmax > global_outline[d*2+1]:
                    global_outline[d*2+1] = xmax

        return global_outline

    def _get_outline_cluster(self, cluster_data):
        """
        returns the outline of the cluster_data

        returns an array
        """

        this_cluster_data_outline = tt.helpers.getMaxOutline(self._ndim)

        for (x, Y) in cluster_data:

            for d in range(self._ndim):
                x = Y[:, d]
                xmin = x.min()
                xmax = x.max()
                if (this_cluster_data_outline[d*2] > xmin):
                    this_cluster_data_outline[d*2] = xmin
                if (this_cluster_data_outline[d*2+1] < xmax):
                    this_cluster_data_outline[d*2+1] = xmax

        return this_cluster_data_outline
