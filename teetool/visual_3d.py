# functions to visualise the information
# (trajectories / probability) in 3 dimensions

import numpy as np
from scipy.interpolate import griddata
import mayavi.mlab as mlab

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

    def plotLogProbability(self, list_clusters, pmin=0.8, pmax=1.0):
        """
        plots log-probability
        """

        [xx, yy, zz] = self._world.getGrid(ndim=3,
                                           resolution=[40, 40, 40])

        ss = np.zeros_like(xx)

        for icluster in list_clusters:
            this_cluster = self._world.getCluster(icluster)
            if ("logp" in this_cluster):
                (Y, s) = this_cluster["logp"]
                s_min = np.min(s)
                # interpolate result
                ss1 = griddata(Y, s, (xx, yy, zz),
                               method='linear',
                               fill_value=s_min)
                # sum
                ss += ss1

        # normalise
        ss_norm = (ss - np.min(ss)) / (np.max(ss) - np.min(ss))

        # mayavi
        src = mlab.pipeline.scalar_field(xx, yy, zz, ss_norm)

        # plot a volume
        mlab.pipeline.volume(src, vmin=pmin, vmax=pmax)
        # slice it
        mlab.pipeline.image_plane_widget(src,
                                         plane_orientation='z_axes',
                                         slice_index=10,
                                         )

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

    def save(self, add=None):
        """
        saves as file
        """

        if (add==None):
            saveas = self._world.getName()
        else:
            saveas = "{0}_{1}".format(self._world.getName(), add)

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
