# functions to visualise the information (trajectories / probability) in 3 dimensions

import mayavi.mlab as mlab
import numpy as np

from teetool import helpers

class Visual_3d(object):
    """
    <description>
    """

    def __init__(self, thisWorld):
        """
        <description>
        """

        # start figure
        self.mfig = mlab.figure()
        self._world = thisWorld

    def plotTrajectories(self, list_clusters):
        """
        <description>
        """

        colours = helpers.getDistinctColours(len(list_clusters))

        for (i, icluster) in enumerate(list_clusters):
            this_cluster = self._world.getCluster(icluster)
            for (x, Y) in this_cluster["data"]:
                mlab.plot3d(Y[:, 0], Y[:, 1], Y[:, 2], color=colours[i], tube_radius=.2)


    def plotLogProbability(self, list_clusters):
        """
        plots log-probability
        """

        [xx, yy, zz] = self._world.getGrid()

        s = np.zeros_like(xx)

        for icluster in list_clusters:
            this_cluster = self._world.getCluster(icluster)
            if ("logp" in this_cluster):
                s += this_cluster["logp"]

        # normalise
        s = (s - np.min(s)) / (np.max(s) - np.min(s))

        # mayavi
        src = mlab.pipeline.scalar_field(xx, yy, zz, s)
        # mlab.pipeline.iso_surface(src, contours=[s.min()+0.3*s.ptp(), ], opacity=0.2)
        mlab.pipeline.volume(src, vmin=.2, vmax=.8)

    def plotOutline(self):
        """
        adds an outline
        """

        outline = self._world.getOutline()

        mlab.outline(extent=outline)


    def show(self):
        """
        shows the image [waits for user input]
        """

        # show figure
        mlab.show()
