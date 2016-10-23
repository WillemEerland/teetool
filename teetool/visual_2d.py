# functions to visualise the information
# (trajectories / probability) in 2 dimensions

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

import teetool as tt


class Visual_2d(object):
    """
    <description>
    """

    def __init__(self, thisWorld, **kwargs):
        """
        <description>
        """

        # start figure
        self._fig = plt.figure(facecolor="white", **kwargs)
        self._ax = self._fig.gca()

        self._ax.set_axis_bgcolor('white')

        self._world = thisWorld

        self._labels = []

    def plotMean(self, list_icluster=None, colour=None, **kwargs):
        """
        plots the mean trajectories
        """

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract data
        clusters = self._world.getCluster(list_icluster)

        # unique colours
        colours = tt.helpers.getDistinctColours(len(clusters), colour)

        clusters = self._world.getCluster(list_icluster)
        for (i, this_cluster) in enumerate(clusters):
            # pass clusters
            Y = this_cluster["model"].getMean()

            a_line, = self._ax.plot(Y[:, 0],
                                    Y[:, 1],
                                    color=colours[i],
                                    **kwargs)




    def plotTrajectories(self, list_icluster=None, ntraj=50,
                         colour=None, **kwargs):
        """
        <description>
        """

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract data
        clusters = self._world.getCluster(list_icluster)

        # unique colours
        colours = tt.helpers.getDistinctColours(len(clusters), colour)

        clusters = self._world.getCluster(list_icluster)
        for (i, this_cluster) in enumerate(clusters):
            # pass clusters
            for itraj, (x, Y) in enumerate(this_cluster["data"]):
                a_line, = self._ax.plot(Y[:, 0],
                                        Y[:, 1],
                                        color=colours[i],
                                        **kwargs)
                # limit number of trajectories
                if itraj > ntraj:
                    break

        self._labels.append((a_line, "data"))

    def plotBox(self, coord_lowerleft, coord_upperright, **kwargs):
        """
        plots box based on two coordinates

        input:
        - coord_lowerleft (x, y)
        - coord_upperright (x, y)
        """

        x_lo = coord_lowerleft[0]
        x_hi = coord_upperright[0]
        y_lo = coord_lowerleft[1]
        y_hi = coord_upperright[1]

        coords = np.array([[x_lo, y_lo],
                           [x_hi, y_lo],
                           [x_hi, y_hi],
                           [x_lo, y_hi],
                           [x_lo, y_lo]])

        coords_x = coords[:,0]
        coords_y = coords[:,1]

        self._ax.plot(coords_x, coords_y, **kwargs)

    def plot(self, *args, **kwargs):
        """
        plotting function - standard matplotlib
        """

        self._ax.plot(*args, **kwargs)

    def plotSamples(self, list_icluster):
        """
        <description>
        """

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        colours = tt.helpers.getDistinctColours(len(list_icluster))

        for (i, icluster) in enumerate(list_icluster):
            these_samples = self._world.getSamples(icluster)
            for (x, Y) in these_samples:
                a_line, = self._ax.plot(Y[:, 0],
                                       Y[:, 1],
                                       color="red",
                                       linestyle=":")

        self._labels.append((a_line, "samples"))

    def plotLegend(self):
        """
        <description>
        """

        list_lines = []
        list_label = []

        for (a_line, a_label) in self._labels:
            list_lines.append(a_line)
            list_label.append(a_label)

        plt.legend(handles=list_lines, labels=list_label)

    def plotTube(self, list_icluster=None, sdwidth=1, z=None, resolution=None):
        """
        plots tube

        list_icluster is a list of lcusters, None is all
        popacity relates to the opacity [0, 1]
        """

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getTube(list_icluster,
                                                      sdwidth,
                                                      z=z,
                                                      resolution=resolution)

        # get colours
        lcolours = tt.helpers.getDistinctColours(len(ss_list))

        for i, ss1 in enumerate(ss_list):
            # plot an iso surface
            plt.contour(xx, yy, ss1, [0.0, 1.0], colors=(lcolours[i], ))

    def plotLogLikelihood(self, list_icluster=None, pmin=0, pmax=1,
                          z=None, resolution=None):
        """
        plots log-likelihood

        input parameters:
            - list_icluster
            - pmin/pmax 0,1
        """

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        (ss_list, [xx, yy, zz]) = self._world.getLogLikelihood(
                                                    list_icluster,
                                                    z=z,
                                                    resolution=resolution)

        ss = np.zeros_like(xx)

        for ss1 in ss_list:
            # sum
            ss += ss1

        # normalise
        ss_norm = (ss - np.min(ss)) / (np.max(ss) - np.min(ss))

        # plot contours
        self._ax.pcolor(xx, yy, ss_norm, cmap="viridis", vmin=pmin, vmax=pmax)


    def plotOutline(self):
        """
        adds an outline

        """

        # TODO draw a box

    def _plotTitle(self):
        """
        adds a title

        """

        # add title
        world_name = self._world.getName()
        if not (world_name == None):
            plt.title(world_name)

    def save(self, add=None):
        """
        saves as file
        """

        if (add==None):
            saveas = self._world.getName()
        else:
            saveas = "{0}_{1}".format(self._world.getName(), add)

        plt.savefig("output/2d_{0}.png".format(saveas))

    def show(self):
        """
        shows the image [waits for user input]
        """



        plt.show()

    def close(self):
        """
        closes the figure
        """

        plt.close("all")
