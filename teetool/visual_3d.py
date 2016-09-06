# functions to visualise the information
# (trajectories / probability) in 3 dimensions

import mayavi.mlab as mlab
import numpy as np
import teetool as tt


class Visual_3d(object):
    """
    <description>
    """

    def __init__(self, thisWorld):
        """
        <description>
        """

        # start figure
        self._mfig = mlab.figure(size=(800,600))
        self._world = thisWorld

    def plotTrajectories(self, list_clusters):
        """
        <description>
        """

        colours = tt.helpers.getDistinctColours(len(list_clusters))

        for (i, icluster) in enumerate(list_clusters):
            this_cluster = self._world.getCluster(icluster)
            for (x, Y) in this_cluster["data"]:
                mlab.plot3d(Y[:, 0], Y[:, 1], Y[:, 2], color=colours[i],
                            tube_radius=None)

    def plotLogProbability(self, list_clusters, pmin=0.5, pmax=1.0):
        """
        plots log-probability
        """

        [xx, yy, zz] = self._world.getGrid(ndim=3)

        s = np.zeros_like(xx)

        for icluster in list_clusters:
            this_cluster = self._world.getCluster(icluster)
            if ("logp" in this_cluster):
                s += this_cluster["logp"]

        # normalise
        s = (s - np.min(s)) / (np.max(s) - np.min(s))

        # mayavi
        src = mlab.pipeline.scalar_field(xx, yy, zz, s)
        mlab.pipeline.iso_surface(src, contours=[.8, .7, .6], opacity=0.2)
        mlab.pipeline.volume(src, vmin=pmin, vmax=pmax)

    def plotOutline(self):
        """
        adds an outline
        """

        outline = self._world.getOutline()

        mlab.outline(extent=outline)

    def _plotTitle(self):
        """
        adds a title
        """

        # add title
        world_name = self._world.getName()

        if not (world_name == None):
            mlab.title(world_name)

    def save(self, saveas=None):
        """
        saves as file
        """

        if (saveas==None):
            saveas = self._world.getName()

        mlab.savefig("output/3d_{0}.png".format(saveas), figure=self._mfig)

    def show(self):
        """
        shows the image [waits for user input]
        """

        # show figure
        mlab.show()

    def close(self):
        """
        closes figure(s)
        """

        mlab.close(all=True)
