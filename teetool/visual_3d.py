# functions to visualise the information (trajectories / probability) in 3 dimensions

import mayavi.mlab as mlab
import numpy as np

from teetool import helpers

def show_trajectories(thisWorld):
    """
    <description>
    """

    clusters = thisWorld.getClusters()

    nclusters = len(clusters)  # number of clusters
    colours = helpers.getDistinctColours(nclusters)  # colours

    mlab.figure()

    for (i, this_cluster) in enumerate(clusters):
        # this cluster
        cluster_data = this_cluster["data"]
        for (x, Y) in cluster_data:
            # this trajectory
            mlab.plot3d(Y[:, 0], Y[:, 1], Y[:, 2], color=colours[i], tube_radius=.1)

    mlab.show()

    return True

def show_intersection(thisWorld, x, y, z):
    """
    <description>
    """

    s = thisWorld.getIntersection(x, y, z)

    # mayavi
    src = mlab.pipeline.scalar_field(s)
    mlab.pipeline.iso_surface(src, contours=[s.min()+0.3*s.ptp(), ], opacity=0.2)
    mlab.pipeline.volume(mlab.pipeline.scalar_field(s), vmin=.2, vmax=.8)
    mlab.outline()
    mlab.show()
