## @package teetool
#  This module contains the main class used in the teetool package
#
#  "main" source file for trajectory analysis in this package. Clusters of trajectories are added here, modelled accordingly (whatever settings desired), and visualised in whatever is desired (single / multiple clusters)

import numpy as np
import teetool as tt
import itertools

## World class handles all trajectory data
#
# This class provides the direct interface to the trajectory data and produce models
class World(object):

    ## initialise World
    # @param self object pointer
    # @param name world name, used in title and saving image
    # @param ndim number of dimensions in trajectory data
    # @param resolution specify resolution of grid
    def __init__(self, name="", ndim=3, resolution=[10, 10, 10]):
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

        ## name of world
        self._name = name
        ## number of dimensions trajectory data
        self._ndim = ndim
        ## list holding clusters
        self._clusters = []
        ## resolution of grid
        self._resolution = resolution
        ## fraction on edges, default 0.1
        self.fraction_to_expand = 0.1

    ## clear previously stored calculations
    # @param self object pointer
    def clear(self):
        for (i, this_cluster) in enumerate(self._clusters):

            if ("model" in this_cluster):
                # clears stored information
                this_cluster["model"].clear()

    ## prints an overview (text)
    # @param self object pointer
    def overview(self):
        print("*** overview [{0}] ***".format(self._name))

        for (i, this_cluster) in enumerate(self._clusters):

            if ("model" in this_cluster):
                has_model = "*"
            else:
                has_model = "-"

            print("{0} [{1}] [{2}]".format(
                        i, this_cluster["name"], has_model))

    ## add a cluster to the world
    # @param self object pointer
    # @param cluster_data list of (x, Y) trajectory data
    # @param cluster_name name of added cluster
    def addCluster(self, cluster_data, cluster_name=""):
        # validate cluster_name
        if type(cluster_name) is not str:
            raise TypeError(
                "expected string, not {0}".format(type(cluster_name)))


        # validate cluster_data
        cluster_data = self._validate_cluster_data(cluster_data)

        # add new cluster [ holds "name" and "data" ]
        new_cluster = {}

        new_cluster["name"] = cluster_name

        # obtain the outline of this data
        outline = tt.helpers.get_cluster_data_outline(cluster_data)
        new_cluster["outl"] = outline

        # store trajectories + time-warping
        new_cluster["data"] = tt.helpers.normalise_data(cluster_data)

        # add cluster to the list
        self._clusters.append(new_cluster)

    ## validates cluster_data
    def _validate_cluster_data(self, cluster_data):

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

            x = np.array(x, dtype=float)
            Y = np.array(Y, dtype=float)

            (M, D) = Y.shape
            if (D != self._ndim):
                raise ValueError("dimension not correct")
            if (M != np.size(x, 0)):
                raise ValueError("number of data-points do not match")
            # check if all finite
            if not np.isfinite(x).all():
                raise ValueError("x holds non-finite values")

            # (over-)write, incase type is not float but int
            trajectory_data = (x, Y)
            cluster_data[i] = trajectory_data

        return cluster_data





    ## returns the name of the world
    # @param self object pointer
    # @return name of the world
    def getName(self):
        if self._name == "":
            return None
        else:
            return self._name

    ## obtain cluster data
    # @param self object pointer
    # @param list_icluster list of clusters to return
    # @return clusters a list of clusters as specified, holds (x, Y)
    def getCluster(self, list_icluster=None):
        # check validity list
        list_icluster = self._check_list_icluster(list_icluster)

        # return clusters in list
        clusters = []
        for i in list_icluster:
            clusters.append(self._clusters[i])
        return clusters


    ## obtain cluster data at specific points [0, 1]
    # @param self object pointer
    # @param x1 timing [0, 1] to visualise points
    # @param list_icluster list of clusters to return
    # @return clusters a list of points A, with points
    def getClusterPoints(self, x1, list_icluster=None):
        # check validity list
        list_icluster = self._check_list_icluster(list_icluster)

        # return clusters in list
        clusters = []
        for i in list_icluster:
            cluster_data = self._clusters[i]["data"]
            A = self._get_point_from_cluster_data(cluster_data, x1)
            clusters.append(A)
        return clusters

    ## obtain statistics of confidence region in scenarios
    # @param self object pointer
    # @param list_icluster list of clusters to analyse
    # @param sdwidth variance to evaluate
    # @param resolution resolution of grid
    def getTubeStats(self, list_icluster=None, sdwidth=1., resolution=None):
        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        # extract
        (ss_list, [xx, yy, zz]) = self.getTube(list_icluster,
                                               sdwidth,
                                               resolution)

        # obtain a list of possible pairs
        list_pairs = []

        ncluster = len(list_icluster)
        temp_list = range(0, ncluster)

        # i1, i2 are counting
        for i1 in temp_list[:-1]:
            for i2 in temp_list[i1+1:]:
                # add tuple
                list_pairs.append((i1, i2))
                # print("* {0} {1}".format(i1, i2))

        for (i1, i2) in list_pairs:
            # to produce

            # these are the actual labels
            i1_id = list_icluster[i1]
            i2_id = list_icluster[i2]

            # 1 :: blocks added
            ss_added = 1.*((ss_list[i1] - ss_list[i2])==-1)

            # 2 :: blocks removed
            ss_removed = 1.*((ss_list[i1] - ss_list[i2])==1)

            # 3 :: present in both
            ss_neutral = 1.*((ss_list[i1] + ss_list[i2])==2)

            # count blocks
            nblocks_add = np.count_nonzero(ss_added)
            nblocks_rem = np.count_nonzero(ss_removed)
            nblocks_neu = np.count_nonzero(ss_neutral)
            # total (of original)
            nblocks_tot = 1.*(nblocks_rem + nblocks_neu)
            # percentages
            nblocks_add_perc = nblocks_add / nblocks_tot
            nblocks_rem_perc = nblocks_rem / nblocks_tot
            nblocks_neu_perc = nblocks_neu / nblocks_tot

            # generate string
            str_output = "{0} vs {1}\nadd/rem/neu\n{2:.3}/{3:.3}/{4:.3}\nadd/rem/neu/tot\n{5}/{6}/{7}/{8}".format(i1_id, i2_id, nblocks_add_perc, nblocks_rem_perc, nblocks_neu_perc,nblocks_add,nblocks_rem,nblocks_neu,nblocks_tot)

            print(str_output)

    ## checks the validity of icluster
    # @param self object pointer
    # @param icluster value to test
    def _check_icluster(self, icluster):
        if type(icluster) is not int:
            raise TypeError("expected integer, not {0}".format(type(icluster)))

        nclusters = len(self._clusters)
        if ((icluster < 0) or (icluster >= nclusters)):
            raise ValueError(
                "{0} not in range [0,{1}]".format(icluster, nclusters))

    ## checks the validity of a list of icluster
    # @param self object pointer
    # @param list_icluster list of icluster to test
    def _check_list_icluster(self, list_icluster):
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

        # TODO finish function?

        return list_inside

    ## returns samples in a list of (x, Y) data
    # @param self object pointer
    # @param icluster specify from which cluster to sample
    # @param nsamples number of samples to generate
    def getSamples(self, icluster, nsamples=50):
        # check validity
        self._check_icluster(icluster)

        # extract
        this_cluster = self._clusters[icluster]

        generated_samples = this_cluster["model"].getSamples(nsamples)

        return generated_samples

    ## generates a model from the trajectory data
    # @param self object pointer
    # @param settings specify settings in dictionary
    # model_type is resampling, ML, or EM
    # ngaus is number of Gaussians
    # @param list_icluster
    def buildModel(self, settings, list_icluster=None):
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

    ## returns the mean trajectory [x, y, z] for list_icluster
    # @param self object pointer
    # @param list_icluster specify clusters
    # @return mean_list a list of means
    def getMean(self, list_icluster=None):
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

    ## returns a grid with bools to specify whether a point falls within the confidence region or not
    # @param self object pointer
    # @param list_icluster which clusters to return
    # @param sdwidth variance to test
    # @param resolution the resolution to produce the grid on
    # @param z (optional) height parameter to produce 2d grids for 3d trajectories
    # @return ss_list a list with ss, representing the bools for each cluster
    # @return xx grid in x-domain
    # @return yy grid in y-domain
    # @return zz grid in z-domain
    def getTube(self,
                list_icluster=None,
                sdwidth=1,
                resolution=None,
                z=None):
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

    ## returns a grid with the maximum log-likelihood
    # @param self object pointer
    # @param list_icluster which clusters to return
    # @param resolution the resolution to produce the grid on
    # @param z (optional) height parameter to produce 2d grids for 3d trajectories
    # @return ss_list a list with ss, representing the bools for each cluster
    # @return xx grid in x-domain
    # @return yy grid in y-domain
    # @return zz grid in z-domain
    def getLogLikelihood(self,
                         list_icluster=None,
                         resolution=None,
                         z=None):
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

    def getComplexityMap(self, list_icluster=None, complexity=1, resolution=None, z=None):

        # check validity
        list_icluster = self._check_list_icluster(list_icluster)

        nclusters = len(list_icluster)

        # obtain log-likelihood
        (ss_list, [xx, yy, zz]) = self.getLogLikelihood(list_icluster,
                                                        resolution,
                                                        z)

        # initialise, flatten
        ss_flat = np.zeros_like(xx, dtype=object).flatten()

        # create empty lists
        for i, _ in enumerate(ss_flat):
            # empty list
            ss_flat[i] = []

        # loop over clusters
        for ss_cluster in ss_list:
            # flatten
            ss_cluster_flat = ss_cluster.flatten()

            # append to list
            for i, _ in enumerate(ss_flat):
                ss_flat[i].append(ss_cluster_flat[i])

        # ### ### ### ### ###
        # ss_flat is now an array that holds a list of logp related to each cluster at every element
        # ### ### ### ### ###

        # now have an array with lists
        for i, _ in enumerate(ss_flat):
            # convert to array
            ss_array_logp = np.array(ss_flat[i])

            # from [logp]robability to [prob]ability
            ss_array_prob = np.exp(ss_array_logp)

            total_prob = 0.0

            for c in np.arange(complexity-1, nclusters)+1:

                # find combinations
                gen = itertools.combinations(np.arange(nclusters), c)

                for these_combinations in gen:

                    # multiply combinations (sum of logp)
                    logp1 = 0.0
                    for this_combination in these_combinations:
                        logp1 += ss_array_logp[this_combination]

                    # convert from [logp]rob to [prob]
                    prob1 = np.exp(logp1)

                    # sum individual probabilities OR
                    total_prob += prob1

            # store value
            # ss_flat[i] = np.log(total_prob)
            ss_flat[i] = total_prob

            """
            # obtain indices sorted
            indices = np.argsort(ss_array_logp)

            # reverse, most likely first
            indices = indices[::-1]

            sum_logp = 0

            # complexity 1, only first value,
            # complexity 2, first two values, etc. etc.
            for c in np.arange(complexity, nclusters):
                # extract index
                idx = indices[c]
                # product probability =
                # sum log probability
                sum_logp += ss_array[idx]

            # store value
            ss_flat[i] = sum_logp
            """

        # convert ss_flat to original structure
        ss = ss_flat.reshape(xx.shape).astype(float)

        return (ss, xx, yy, zz)



    ## returns a grid based on outline and resolution
    # @param self object pointer
    # @param outline outline to generate grid from
    # @param resolution resolution to generate grid from
    # @return xx grid in x-domain
    # @return yy grid in y-domain
    # @return zz grid in z-domain
    def _getGrid(self, outline, resolution=None):
        # default resolution
        if resolution is None:
            resolution = self._resolution

        # use expanded grid for calculations
        #outline = self._get_outline_expanded(list_icluster)

        [xx, yy, zz] = tt.helpers.getGridFromResolution(outline, resolution)

        return [xx, yy, zz]

    ## returns an expanded grid based on the outline of trajectory data in list_icluster
    # @param self object pointer
    # @param list_icluster list of clusters to take into account
    # @param fraction_to_expand fraction to expand the outline with
    # @return xx grid in x-domain
    # @return yy grid in y-domain
    # @return zz grid in z-domain
    def _get_outline_expanded(self, list_icluster=None, fraction_to_expand=0.1):
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

    ## returns a grid based on the outline of trajectory data in list_icluster
    # @param self object pointer
    # @param list_icluster list of clusters to take into account
    # @return xx grid in x-domain
    # @return yy grid in y-domain
    # @return zz grid in z-domain
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

    ## returns the outline of a tube of specified clusters
    # @param self object pointer
    # @param sdwidth variance to evaluate confidence region on
    # @param list_icluster clusters to evaluate confidence on. None shows all
    # @return global_outline global outline of the tube (maximum dimensions)
    def _get_outline_tube(self, sdwidth=1, list_icluster=None):
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


    def _get_point_from_cluster_data(self, cluster_data, x1):

        # obtain values
        A = []

        for (x, Y) in cluster_data:

            # obtain point
            a = self._get_point_from_xY(x, Y, x1)

            A.append(a)

        # from list to array
        A = np.array(A).squeeze()

        return A

    ## returns the points nearest to x1
    # @param self object pointer
    # @param x x from (x, Y), to find the relevant index
    # @param Y Y from (x, Y), one of these values gets returned
    # @param x1 timing [0, 1] to visualise points
    # @return a a point in space, along the position x, value Y
    def _get_point_from_xY(self, x, Y, x1):

        # obtain index
        idx = np.argmin( np.abs(x-x1) )

        a = Y[idx,:]

        return a
