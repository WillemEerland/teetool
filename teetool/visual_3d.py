# functions to visualise the information (trajectories / probability) in 3 dimensions

import mayavi.mlab as mlab
import numpy as np

from teetool import helpers

class Visual_3d(object):
    """
    <description>
    """

    def __init__(self):
        """
        <description>
        """

        # start figure
        self.mfig = mlab.figure()

        # initial outline
        self._outline = [np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf]


    def add_trajectories(self, thisWorld):
        """
        <description>
        """

        clusters = thisWorld.getClusters()

        nclusters = len(clusters)  # number of clusters
        colours = helpers.getDistinctColours(nclusters)  # colours

        for (i, this_cluster) in enumerate(clusters):
            # this cluster
            cluster_data = this_cluster["data"]
            for (x, Y) in cluster_data:
                # this trajectory
                y0 = Y[:, 0]
                y1 = Y[:, 1]
                y2 = Y[:, 2]
                mlab.plot3d(y0, y1, y2, color=colours[i], tube_radius=.1)
                # outline
                self._check_outline([y0, y1, y2])


        return True


    def add_intersection(self, thisWorld, x, y, z):
        """
        <description>
        """

        # outline
        self._check_outline([x, y, z])

        s = thisWorld.getIntersection(x, y, z)

        # mayavi
        src = mlab.pipeline.scalar_field(x, y, z, s)
        # mlab.pipeline.iso_surface(src, contours=[s.min()+0.3*s.ptp(), ], opacity=0.2)
        mlab.pipeline.volume(src, vmin=.2, vmax=.8)

    def show(self):
        """
        <description>
        """

        # add outline TODO add extent=[xmin, xmax, ymin, ymax, zmin, zmax]
        mlab.outline(extent=[-60, 60, -10, 240, -60, 60])

        # show figure
        mlab.show()

    def _check_outline(self, xyz):
        """
        calculate maximum outline from list and store
        """

        for d in range(3):
            x = xyz[d]
            xmin = x.min()
            xmax = x.max()
            if (self._outline[d*2] > xmin):
                self._outline[d*2] = xmin
            if (self._outline[d*2+1] > xmax):
                self._outline[d*2+1] = xmax
