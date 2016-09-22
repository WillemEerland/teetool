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

    def plotTrajectories(self, list_icluster=None, linewidth=2, reduced=30):
        """
        plot trajectories

        list_icluster should be a list of clusters, integers only
        """

        # extract data
        clusters = self._world.getCluster(list_icluster)

        # unique colours
        colours = tt.helpers.getDistinctColours(len(clusters))

        for (i, this_cluster) in enumerate(clusters):
            # pass clusters
            for (x, Y) in this_cluster["data"]:
                #mlab.plot3d(Y[:, 0], Y[:, 1], Y[:, 2], color=colours[i])
                (npoints, ndim) = Y.shape
                # reduce points to 30
                step_size = int(npoints / reduced)
                these_rows = range(0, npoints, step_size)
                Y_red = Y[these_rows,:]
                mlab.plot3d(Y_red[:, 0], Y_red[:, 1], Y_red[:, 2], color=colours[i], line_width=linewidth)

    def plotLogDifference(self, icluster1, icluster2, pmin=0.0, pmax=1.0, popacity=0.3):
        """
        plots difference

        icluster1, and icluster2 should be both integers
        """

        (ss_list, [xx, yy, zz]) = self._world.getLogLikelihood([icluster1, icluster2])

        ss = np.zeros_like(xx)

        # add
        ss += ss_list[0]

        # remove
        ss -= ss_list[1]

        # normalise
        ss_norm = (ss - np.min(ss)) / (np.max(ss) - np.min(ss))

        # show peak areas
        mlab.pipeline.iso_surface(src, contours=[pmin, pmax], opacity=0.1)

        # slice it
        mlab.pipeline.image_plane_widget(src,
                                         plane_orientation='z_axes',
                                         slice_index=10,
                                         vmin=pmin,
                                         vmax=pmax)

        # mayavi
        src = mlab.pipeline.scalar_field(xx, yy, zz, ss_norm)

        # slice it
        mlab.pipeline.image_plane_widget(src,
                                         plane_orientation='z_axes',
                                         slice_index=10,
                                         )

    def plotTubeDifference(self, icluster1, icluster2, sdwidth=1, popacity=0.1):
        """
        plots difference between sets

        input parameters:
            - icluster1
            - icluster2
            - sdwidth
            - popacity
        """

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getTube([icluster1, icluster2], sdwidth)


        for i in range(3):
            # 3 different cases

            if i==1:
                # 1 :: blocks added
                ss1 = 1*((ss_list[0] - ss_list[1])==1)
                colour = (0.0, 1.0, 0.0)  # green
                label = "added"
            elif i==2:
                # 2 :: blocks removed
                ss1 = 1*((ss_list[0] - ss_list[1])==-1)
                colour = (1.0, 0.0, 0.0)  # red
                label = "removed"
            else:
                # 3 :: present in both
                ss1 = 1*((ss_list[0] + ss_list[1])==2)
                colour = (0.0, 0.0, 1.0)  # blue
                label = "neutral"

            #
            src = mlab.pipeline.scalar_field(xx, yy, zz, ss1)
            #
            mlab.pipeline.iso_surface(src,
                                      contours=[0.5],
                                      opacity=popacity,
                                      color=colour,
                                      name=label)

        # slice it
        """
        src = mlab.pipeline.scalar_field(xx, yy, zz, ss)
        mlab.pipeline.image_plane_widget(src,
                                       plane_orientation='z_axes',
                                       slice_index=10,
                                       vmin=0.4,
                                       vmax=0.6)
                                       """

    def plotTube(self, list_icluster=None, sdwidth=1, popacity=0.3):
        """
        plots log-probability

        list_icluster is a list of lcusters, None is all
        popacity relates to the opacity [0, 1]
        """

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getTube(list_icluster, sdwidth)

        # get colours
        lcolours = tt.helpers.getDistinctColours(len(ss_list))

        for i, ss1 in enumerate(ss_list):

            # mayavi
            src = mlab.pipeline.scalar_field(xx, yy, zz, ss1)

            # plot an iso surface
            mlab.pipeline.iso_surface(src,
                                      contours=[0.5],
                                      opacity=popacity,
                                      color=lcolours[i])

    def plotLogLikelihood(self, list_icluster=None, pmin=0.0, pmax=1.0, popacity=0.3):
        """
        plots log-likelihood

        input parameters:
            - list_icluster
            - pmin
            - pmax
        """

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getLogLikelihood(list_icluster)

        ss = np.zeros_like(xx)

        for ss1 in ss_list:
            # sum
            ss += ss1

        # normalise
        ss_norm = (ss - np.min(ss)) / (np.max(ss) - np.min(ss))

        # mayavi
        src = mlab.pipeline.scalar_field(xx, yy, zz, ss_norm)

        # show peak areas
        mlab.pipeline.iso_surface(src, contours=[pmin, pmax], opacity=0.1)
        # plot a volume
        #mlab.pipeline.volume(src, vmin=pmin, vmax=pmax)
        # slice it
        mlab.pipeline.image_plane_widget(src,
                                         plane_orientation='z_axes',
                                         slice_index=10,
                                         vmin=pmin,
                                         vmax=pmax)

    def plotOutline(self, list_icluster=None):
        """
        adds an outline

        input parameters:
            - list_icluster
        """

        plot_outline = self._world._get_outline(list_icluster)

        mlab.outline(extent=plot_outline)

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
