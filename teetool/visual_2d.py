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

        [xmin, xmax, ymin, ymax] = thisWorld.getOutline()
        self._ax.set_xlim([xmin, xmax])
        self._ax.set_ylim([ymin, ymax])

        self._world = thisWorld
        self._plotTitle()

        self._labels = []

    def plotTrajectories(self, list_clusters):
        """
        <description>
        """

        colours = tt.helpers.getDistinctColours(len(list_clusters))

        for (i, icluster) in enumerate(list_clusters):
            this_cluster = self._world.getCluster(icluster)
            for (x, Y) in this_cluster["data"]:
                a_line, = self._ax.plot(Y[:, 0],
                                       Y[:, 1],
                                       color="black",
                                       linestyle="-")

        self._labels.append((a_line, "data"))

    def plotSamples(self, list_clusters):
        """
        <description>
        """

        colours = tt.helpers.getDistinctColours(len(list_clusters))

        for (i, icluster) in enumerate(list_clusters):
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

    def plotLogProbability(self, list_clusters, pmin=0, pmax=1):
        """
        plots log-probability
        ncontours: number of contours drawn
        """

        [xx, yy] = self._world.getGrid(ndim=2,
                                       resolution=[100, 100])

        ss = np.zeros_like(xx)

        for icluster in list_clusters:
            this_cluster = self._world.getCluster(icluster)
            if ("logp" in this_cluster):
                (Y, s) = this_cluster["logp"]
                s_min = np.min(s)
                # interpolate result
                ss1 = griddata(Y, s, (xx, yy),
                               method='linear',
                               fill_value=s_min)
                # sum
                ss += ss1

        # normalise
        ss_norm = (ss - np.min(ss)) / (np.max(ss) - np.min(ss))

        # plot contours
        self._ax.pcolor(xx, yy, ss_norm, cmap="viridis", vmin=pmin, vmax=pmax)
        #self._ax.contourf(xx, yy, ss_norm, ncontours, cmap="viridis")

    def plotOutline(self):
        """
        adds an outline
        """

        # TODO draw a box

        return True

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
