## @package teetool
#  This module contains the Visual_3d class
#
#  See Visual_3d class for more details

import numpy as np
from scipy.interpolate import griddata
import mayavi.mlab as mlab
import time

import teetool as tt

## Visual_3d class generates the 3d output using Mayavi
class Visual_3d(object):

    ## Constructor for Visual_3d
    # @param self object pointer
    # @param thisWorld World object, filled with trajectory data and models
    # @param kwargs additional parameters for mlab.figure()
    def __init__(self, thisWorld, **kwargs):
        ## Mayavi figure
        self._mfig = mlab.figure(bgcolor=(1.0, 1.0, 1.0),
                                 fgcolor=(0.0, 0.0, 0.0),
                                 **kwargs);
        ## World object
        self._world = thisWorld

    ## standard plotting function for Mayavi, plot3d
    # @param self object pointer
    # @param args additional arguments for plotting
    # @param kwargs additional labeled parameters for plotting
    def plot(self, *args, **kwargs):
        mlab.plot3d(*args, **kwargs)

    ## Plot mean of trajectories
    # @param self object pointer
    # @param list_icluster list of clusters to plot
    # @param colour if specified, overwrites distinct colours
    # @param kwargs additional parameters for plotting
    def plotMean(self, list_icluster=None, colour=None, **kwargs):
        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract data
        clusters = self._world.getCluster(list_icluster)

        # unique colours
        colours = tt.helpers.getDistinctColours(len(self._world._clusters),
                                                 colour)

        for (i, this_cluster) in enumerate(clusters):
            # pass clusters
            Y = this_cluster["model"].getMean()

            mlab.plot3d(Y[:, 0], Y[:, 1], Y[:, 2], color=colours[i],
                        tube_radius=None, **kwargs)

    ## Plot trajectories of cluster
    # @param self object pointer
    # @param list_icluster list of clusters to plot
    # @param ntraj maximum number of trajectories
    # @param colour if specified, overwrites distinct colours
    # @param kwargs additional parameters for plotting
    def plotTrajectories(self, list_icluster=None,
                         ntraj=50, colour=None, **kwargs):
        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract data
        clusters = self._world.getCluster(list_icluster)

        # unique colours
        colours = tt.helpers.getDistinctColours(len(self._world._clusters),
                                                colour)

        for (i, this_cluster) in enumerate(clusters):
            # pass clusters
            for itraj, (x, Y) in enumerate(this_cluster["data"]):

                # limit number of trajectories printed
                if itraj > (ntraj-1):
                    break

                mlab.plot3d(Y[:, 0], Y[:, 1], Y[:, 2], color=colours[i],
                            tube_radius=None, **kwargs)

    ## Plot samples of model
    # @param self object pointer
    # @param list_icluster list of clusters to plot
    # @param ntraj number of trajectories
    # @param colour if specified, overwrites distinct colours
    # @param kwargs additional parameters for plotting
    def plotSamples(self, list_icluster=None, ntraj=50, colour=None, **kwargs):

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # unique colours
        colours = tt.helpers.getDistinctColours(len(self._world._clusters),
                                                 colour)

        for (i, icluster) in enumerate(list_icluster):
            these_samples = self._world.getSamples(icluster,
                                                   nsamples=ntraj)
            for (x, Y) in these_samples:

                mlab.plot3d(Y[:, 0], Y[:, 1], Y[:, 2], color=colours[i],
                            tube_radius=None, **kwargs)

    ## Plot points in trajectories of cluster
    # @param self object pointer
    # @param x1 timing [0, 1] to visualise points
    # @param list_icluster list of clusters to plot
    # @param ntraj maximum number of trajectories
    # @param colour if specified, overwrites distinct colours
    # @param kwargs additional parameters for plotting
    def plotTrajectoriesPoints(self, x1, list_icluster=None,
                         ntraj=50, colour=None, **kwargs):
        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # obtain points
        clustersP = self._world.getClusterPoints(x1, list_icluster)

        # unique colours
        colours = tt.helpers.getDistinctColours(len(self._world._clusters),
                                                colour)

        for (i, A) in enumerate(clustersP):
            # pass clusters
            for itraj, a in enumerate(A):

                # limit number of trajectories printed
                if itraj > (ntraj-1):
                    break

                mlab.points3d(a[0], a[1], a[2], color=colours[i],**kwargs)



    def plotLogDifference(self,
                          icluster1,
                          icluster2,
                          pmin=0.0, pmax=1.0):

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

    def plotTube(self, list_icluster=None, sdwidth=1, alpha=1.0,
                  resolution=None, colour=None, **kwargs):
        """
        plots log-probability

        list_icluster is a list of lcusters, None is all
        alpha relates to the opacity [0, 1]
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
                                   opacity=alpha,
                                   color=lcolours[list_icluster[i]],
                                   **kwargs)

    def plotTubeDifference(self, list_icluster=None, sdwidth=1, alpha=1.0,
                           resolution=None, **kwargs):
        """
        plots difference between sets

        input parameters:
            - icluster1
            - icluster2
            - sdwidth
            - alpha
        """

        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract first two only!
        list_icluster = list_icluster[:2]

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getTube(list_icluster,
                                                      sdwidth, resolution)

        # 1 :: blocks added
        ss_added = ((ss_list[0] - ss_list[1])==-1)

        # 2 :: blocks removed
        ss_removed = ((ss_list[0] - ss_list[1])==1)

        # 3 :: present in both
        ss_neutral = ((ss_list[0] + ss_list[1])==2)

        for i in [1, 2, 3]:
            if i == 1:
                ss1 = 1.*ss_removed
                color = (1.0, 0.1, 0.1)
                label = "removed"
            elif i == 2:
                ss1 = 1.*ss_added
                color = (0.1, 1.0, 0.1)
                label = "added"
            elif i == 3:
                ss1 = 1.*ss_neutral
                color = (0.1, 0.1, 1.0)
                label = "neutral"
            # mayavi
            src = mlab.pipeline.scalar_field(xx, yy, zz, ss1)
            # plot an iso surface
            mlab.pipeline.iso_surface(src,
                                   contours=[0.5],
                                   opacity=alpha,
                                   color=color, **kwargs)
            # some stats
            #nblocks_used = np.count_nonzero(ss1)
            #nblocks_total = np.prod(ss1.shape)
            #print("{0} > {1} of {2}".format(label, nblocks_used, nblocks_total))

    ## passes arguments to view
    # @param self object pointer
    # @param kwargs optional keyword parmeters
    # azimuth float, optional. The azimuthal angle (in degrees, 0-360), i.e. the angle subtended by the position vector on a sphere projected on to the x-y plane with the x-axis.
    # elevation float, optional. The zenith angle (in degrees, 0-180), i.e. the angle subtended by the position vector and the z-axis.
    # distance float or auto, optional. A positive floating point number representing the distance from the focal point to place the camera. New in Mayavi 3.4.0: if auto is passed, the distance is computed to have a best fit of objects in the frame.
    # focalpoint array_like or auto, optional. An array of 3 floating point numbers representing the focal point of the camera. New in Mayavi 3.4.0: if auto is passed, the focal point is positioned at the center of all objects in the scene.
    # roll float, optional Controls the roll, ie the rotation of the camera around its axis.
    # reset_roll boolean, optional. If True, and roll is not specified, the roll orientation of the camera is reset.
    # figure The Mayavi figure to operate on. If None is passed, the current one is used.
    def setView(self, **kwargs):
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



    def plotLogLikelihood(self, list_icluster=None,
                            pmin=0.0, pmax=1.0,
                            alpha=0.3,
                            resolution=None):
        """
        plots log-likelihood

        input parameters:
            - list_icluster
            - complexity
        """

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getLogLikelihood(list_icluster,
                                                               resolution)

        ss = ss_list[0] # initialise

        for ss1 in ss_list:
            # find those greater
            mask = np.greater(ss1, ss)
            # replace
            ss[mask] = ss1[mask]

        # normalise
        ss_norm = (ss - np.min(ss)) / (np.max(ss) - np.min(ss))

        # mayavi
        src = mlab.pipeline.scalar_field(xx, yy, zz, ss_norm)

        # show peak areas
        mlab.pipeline.iso_surface(src, contours=[pmin, pmax], opacity=alpha)
        # plot a volume
        #mlab.pipeline.volume(src, vmin=pmin, vmax=pmax)
        # slice it
        mlab.pipeline.image_plane_widget(src,
                                         plane_orientation='z_axes',
                                         slice_index=10,
                                         vmin=pmin,
                                         vmax=pmax)

    def plotComplexityMap(self, list_icluster=None, complexity=1, pmin=0.0, pmax=1.0, alpha=0.3, resolution=None):
        """
        Plot complexity map
        """

        ss, xx, yy, zz = self._world.getComplexityMap(list_icluster,
                                                      complexity,
                                                      resolution)

        # normalise
        ss_norm = (ss - np.min(ss)) / (np.max(ss) - np.min(ss))

        # mayavi
        src = mlab.pipeline.scalar_field(xx, yy, zz, ss_norm)

        # show peak areas
        mlab.pipeline.iso_surface(src, contours=[pmin, pmax], opacity=alpha)
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


    def save(self, add=None, path="output"):
        """
        saves as file
        """

        if (add==None):
            saveas = self._world.getName()
        else:
            saveas = "{0}_{1}".format(self._world.getName(), add)

        #
        mlab.savefig("{0}/3d_{1}.png".format(path, saveas), figure=self._mfig)


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
