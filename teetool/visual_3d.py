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
        self._mfig = mlab.figure(size=(800,600),
                                 bgcolor=(1.0, 1.0, 1.0),
                                 fgcolor=(0.0, 0.0, 0.0))
        self._world = thisWorld

    def plotTrajectories(self, list_icluster=None,
                         ntraj=50, linewidth=1, colour=None):
        """
        plot trajectories

        list_icluster should be a list of clusters, integers only
        """

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract data
        clusters = self._world.getCluster(list_icluster)

        # unique colours
        colours = tt.helpers.getDistinctColours(len(clusters), colour)

        for (i, this_cluster) in enumerate(clusters):
            # pass clusters
            for itraj, (x, Y) in enumerate(this_cluster["data"]):
                mlab.plot3d(Y[:, 0], Y[:, 1], Y[:, 2], color=colours[i],
                            tube_radius=linewidth)

                # limit number of trajectories printed
                if itraj > ntraj:
                    break

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

    def setView(self, **kwargs):
        """
        passes arguments to view

        mayavi.mlab.view(azimuth=None, elevation=None, distance=None, focalpoint=None, roll=None, reset_roll=True, figure=None)

        azimuth:	float, optional. The azimuthal angle (in degrees, 0-360), i.e. the angle subtended by the position vector on a sphere projected on to the x-y plane with the x-axis.
        elevation:	float, optional. The zenith angle (in degrees, 0-180), i.e. the angle subtended by the position vector and the z-axis.
        distance:	float or ‘auto’, optional. A positive floating point number representing the distance from the focal point to place the camera. New in Mayavi 3.4.0: if ‘auto’ is passed, the distance is computed to have a best fit of objects in the frame.
        focalpoint:	array_like or ‘auto’, optional. An array of 3 floating point numbers representing the focal point of the camera. New in Mayavi 3.4.0: if ‘auto’ is passed, the focal point is positioned at the center of all objects in the scene.
        roll:	float, optional Controls the roll, ie the rotation of the camera around its axis.
        reset_roll:	boolean, optional. If True, and ‘roll’ is not specified, the roll orientation of the camera is reset.
        figure:	The Mayavi figure to operate on. If None is passed, the current one is used.
        """

        view = mlab.view(**kwargs)


        return view

    def setLabels(self, xlabel="", ylabel="", zlabel=""):
        """
        sets the label

        input:
            - xlabel
            - ylabel
            - zlabel
        """

        mlab.xlabel(xlabel)
        mlab.ylabel(ylabel)
        mlab.zlabel(zlabel)

    def setAxesFormat(self, newFormat="%.0f"):
        """
        changes the format of axis

        input:
            - newFormat
        """

        # change scientific notation to normal
        ax = mlab.axes()
        ax.axes.label_format = newFormat


    def plotGrid(self, list_icluster=None, resolution=1, outline=None):
        """
        plots a gridplane, based on src

        input:
            - list_icluster
            - resolution
        """

        # obtain an outline
        if outline is None:
            outline = self._world._get_outline(list_icluster)

        # 3d
        [xx, yy, zz] = tt.helpers.getGridFromResolution(outline, resolution)

        # fake data (not used)
        ss = np.ones_like(xx)

        src = mlab.pipeline.scalar_field(xx, yy, zz, ss)

        gx = mlab.pipeline.grid_plane(src)
        gy = mlab.pipeline.grid_plane(src)
        gy.grid_plane.axis = 'y'
        gz = mlab.pipeline.grid_plane(src)
        gz.grid_plane.axis = 'z'

    def plotTube(self, list_icluster=None, sdwidth=1, popacity=0.3,
                 resolution=None, colour=None):
        """
        plots log-probability

        list_icluster is a list of lcusters, None is all
        popacity relates to the opacity [0, 1]
        resolution does the grid
        """

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getTube(list_icluster,
                                                      sdwidth, resolution)

        # get colours
        lcolours = tt.helpers.getDistinctColours(len(self._world._clusters),
                                                 colour)

        for i, ss1 in enumerate(ss_list):

            # mayavi
            src = mlab.pipeline.scalar_field(xx, yy, zz, ss1)

            # plot an iso surface
            mlab.pipeline.iso_surface(src,
                                      contours=[0.5],
                                      opacity=popacity,
                                      color=lcolours[list_icluster[i]])

    def plotLogLikelihood(self, list_icluster=None, pmin=0.0, pmax=1.0,
                          popacity=0.3, resolution=None):
        """
        plots log-likelihood

        input parameters:
            - list_icluster
            - pmin
            - pmax
        """

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getLogLikelihood(list_icluster,
                                                               resolution)

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

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        plot_outline = self._world._get_outline(list_icluster)

        mlab.outline(extent=plot_outline)

    def plotTitle(self, title=None):
        """
        adds a title
        """

        # add title
        if title is None:
            title = self._world.getName()

        mlab.title(title)


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
