# functions to visualise the information
# (trajectories / probability) in 2 dimensions

import numpy as np
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
        self._fig = plt.figure()
        self._ax = self._fig.gca()
        self._world = thisWorld

    def plotTrajectories(self, list_clusters):
        """
        <description>
        """

        colours = tt.helpers.getDistinctColours(len(list_clusters))

        for (i, icluster) in enumerate(list_clusters):
            this_cluster = self._world.getCluster(icluster)
            for (x, Y) in this_cluster["data"]:
                self._ax.plot(Y[:, 0], Y[:, 1], color=colours[i])

    def plotLogProbability(self, list_clusters, ncontours=20):
        """
        plots log-probability
        ncontours: number of contours drawn
        """

        [xx, yy] = self._world.getGrid()

        s = np.zeros_like(xx)

        for icluster in list_clusters:
            this_cluster = self._world.getCluster(icluster)
            if ("logp" in this_cluster):
                s += this_cluster["logp"]

        # normalise
        s = (s - np.min(s)) / (np.max(s) - np.min(s))

        # plot contours
        self._ax.contourf(xx, yy, s, ncontours)

    def plotOutline(self):
        """
        adds an outline
        """

        # TODO draw a box

        return True

    def show(self):
        """
        shows the image [waits for user input]
        """

        plt.show()

    def close(self):
        """
        closes the figure
        """

        plt.close()
