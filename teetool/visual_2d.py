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

    def __init__(self, thisWorld):
        """
        <description>
        """

        # start figure
        self._fig = plt.figure(facecolor="white")
        self._ax = self._fig.gca()

        self._ax.set_axis_bgcolor('grey')

        #[xmin, xmax, ymin, ymax] = thisWorld._get_outline_expanded()
        #self._ax.set_xlim([xmin, xmax])
        #self._ax.set_ylim([ymin, ymax])

        self._world = thisWorld
        self._plotTitle()

        self._labels = []

    def plotTrajectories(self, list_icluster, ntraj=50, bBlack=False):
        """
        <description>
        """

        colours = tt.helpers.getDistinctColours(len(list_icluster), bBlack)

        clusters = self._world.getCluster(list_icluster)
        for (i, this_cluster) in enumerate(clusters):
            # pass clusters
            for itraj, (x, Y) in enumerate(this_cluster["data"]):
                a_line, = self._ax.plot(Y[:, 0],
                                       Y[:, 1],
                                       color="black",
                                       linestyle="-")
                # limit number of trajectories
                if itraj > ntraj:
                    break

        self._labels.append((a_line, "data"))

    def plotSamples(self, list_icluster):
        """
        <description>
        """

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

    def plotTube(self, list_icluster=None, sdwidth=1):
        """
        plots tube

        list_icluster is a list of lcusters, None is all
        popacity relates to the opacity [0, 1]
        """

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getTube(list_icluster, sdwidth)

        # get colours
        lcolours = tt.helpers.getDistinctColours(len(ss_list))

        for i, ss1 in enumerate(ss_list):

            # plot an iso surface
            plt.contour(xx, yy, ss1, [0.0, 1.0], colors=(lcolours[i], ))
            #mlab.pipeline.iso_surface(src,
            #                          contours=[0.5],
            #                          opacity=popacity,
            #                          color=lcolours[i])

    def plotLogLikelihood(self, list_icluster=None, pmin=0, pmax=1):
        """
        plots log-likelihood

        input parameters:
            - list_icluster
            - pmin/pmax 0,1
        """

        (ss_list, [xx, yy, zz]) = self._world.getLogLikelihood(list_icluster)

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
