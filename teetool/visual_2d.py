## @package teetool
#  This module contains the Visual_2d class
#
#  See Visual_2d class for more details

import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

import teetool as tt

## Visual_2d class generates the 2d output using Matplotlib
#
#  Even 3-dimensional trajectories can be output in 2d (sliced)
class Visual_2d(object):

    ## Constructor for Visual_2d
    # @param self object pointer
    # @param thisWorld World object, filled with trajectory data and models
    # @param kwargs additional parameters for plt.figure()
    def __init__(self, thisWorld, **kwargs):
        """
        <description>
        """

        ## figure object
        self._fig = plt.figure(facecolor="white", **kwargs)
        ## axis object
        self._ax = self._fig.gca()
        # set colour of axis
        #self._ax.set_axis_bgcolor('white')
        #self._ax.set_facecolor('white')
        ## World object
        self._world = thisWorld
        ## Labels of plots
        self._labels = []

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

            a_line, = self._ax.plot(Y[:, 0],
                                    Y[:, 1],
                                    color=colours[list_icluster[i]],
                                    **kwargs)

    ## Plot trajectories of cluster
    # @param self object pointer
    # @param list_icluster list of clusters to plot
    # @param ntraj maximum number of trajectories
    # @param colour if specified, overwrites distinct colours
    # @param kwargs additional parameters for plotting
    def plotTrajectories(self,
                         list_icluster=None,
                         ntraj=50,
                         colour=None,
                         **kwargs):
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
                a_line, = self._ax.plot(Y[:, 0],
                                        Y[:, 1],
                                        color=colours[i],
                                        **kwargs)
                # limit number of trajectories
                if itraj > ntraj:
                    break

        self._labels.append((a_line, "data"))

    ## Plot trajectories of cluster
    # @param self object pointer
    # @param x1 point from [0,1] to visualise
    # @param list_icluster list of clusters to plot
    # @param ntraj maximum number of trajectories
    # @param colour if specified, overwrites distinct colours
    # @param kwargs additional parameters for plotting
    def plotTrajectoriesPoints(self,
                               x1,
                               list_icluster=None,
                               ntraj=50,
                               colour=None,
                               **kwargs):
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
                a_line, = self._ax.plot(a[0],
                                        a[1],
                                        color=colours[i],
                                        **kwargs)
                # limit number of trajectories
                if itraj > ntraj:
                    break

        self._labels.append((a_line, "data"))

    ## Plot time-series of trajectories
    # @param self object pointer
    # @param icluster select cluster to plot
    # @param ntraj maximum number of trajectories
    # @param colour specificy colour of trajectories
    # @param kwargs additional parameters for plotting
    def plotTimeSeries(self, icluster=0, ntraj=50,
                         colour='k', **kwargs):
        # number of subplots, 2 or 3
        ndim = self._world._ndim

        # subplot
        f, axarr = plt.subplots(ndim, sharex=True)

        # check validity
        [icluster] = self._world._check_list_icluster([icluster])

        # extract data
        clusters = self._world.getCluster([icluster])
        for (i, this_cluster) in enumerate(clusters):
            # pass clusters
            for itraj, (x, Y) in enumerate(this_cluster["data"]):

                for d in range(ndim):
                    x_norm = (x - x.min()) / (x.max() - x.min())
                    axarr[d].plot(x_norm, Y[:,d],color=colour, **kwargs)

                if itraj > ntraj:
                    break

    ## Plot a box based on two coordinates
    # @param self object pointer
    # @param coord_lowerleft lower-left coordinate (x,y)
    # @param coord_upperright upper-right coordinate (x,y)
    # @param kwargs additional parameters for plotting
    def plotBox(self, coord_lowerleft, coord_upperright, **kwargs):
        x_lo = coord_lowerleft[0]
        x_hi = coord_upperright[0]
        y_lo = coord_lowerleft[1]
        y_hi = coord_upperright[1]

        coords = np.array([[x_lo, y_lo],
                           [x_hi, y_lo],
                           [x_hi, y_hi],
                           [x_lo, y_hi],
                           [x_lo, y_lo]])

        coords_x = coords[:,0]
        coords_y = coords[:,1]

        self._ax.plot(coords_x, coords_y, **kwargs)

    ## standard plotting function for Matplotlib
    # @param self object pointer
    # @param args additional arguments for plotting
    # @param kwargs additional labeled parameters for plotting
    def plot(self, *args, **kwargs):
        # plot
        self._ax.plot(*args, **kwargs)

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
                a_line, = self._ax.plot(Y[:, 0],
                                       Y[:, 1],
                                       color=colours[i],
                                       linestyle=":",
                                       **kwargs)

        self._labels.append((a_line, "samples"))

    ## Add legend to plot
    # @param self object pointer
    def plotLegend(self):
        list_lines = []
        list_label = []

        for (a_line, a_label) in self._labels:
            list_lines.append(a_line)
            list_label.append(a_label)

        plt.legend(handles=list_lines, labels=list_label)

    ## Plots a confidence region of variance sigma
    # @param self object pointer
    # @param list_icluster list of clusters to plot
    # @param sdwidth variance to evaluate
    # @param z if specified, it evaluates the confidence region at a constant altitude for 3D trajectories
    # @param resolution sets resolution for which to calculate the tube, can be a single integer, or an actual measurement [dim1 dim2] (2d) [dim1 dim2 dim3] (3d)
    # @param colour if specified, overwrites distinct colours
    # @param alpha opacity for the confidence region
    # @param kwargs additional parameters for plotting
    def plotTube(self,
                 list_icluster=None,
                 sdwidth=1,
                 z=None,
                 resolution=None,
                 colour=None,
                 alpha=.1,
                 **kwargs):
        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getTube(list_icluster,
                                                      sdwidth,
                                                      z=z,
                                                      resolution=resolution)

        # unique colours
        lcolours = tt.helpers.getDistinctColours(len(self._world._clusters),
                                                 colour)

        for i, ss1 in enumerate(ss_list):
            #plt.contourf(xx, yy, 1.*ss1, levels=[-np.inf, 1., np.inf], colors=(lcolours[i],), alpha=alpha, **kwargs)
            # plot an iso surface line
            plt.contour(xx,
                        yy,
                        ss1,
                        levels=[.5],
                        colors=(lcolours[list_icluster[i]], 'w'),
                        **kwargs)

    ## Plots the difference confidence region of variance sigma for two models
    # @param self object pointer
    # @param list_icluster list of 2 clusters to compare
    # @param sdwidth variance to evaluate
    # @param z if specified, it evaluates the confidence region at a constant altitude for 3D trajectories
    # @param resolution specify resolution of region
    # @param colour if specified, overwrites distinct colours
    # @param alpha opacity for the confidence region
    # @param kwargs additional parameters for plotting
    def plotTubeDifference(self,
                           list_icluster=None,
                           sdwidth=1,
                           z=None,
                           resolution=None,
                           colour=None,
                           alpha=.1,
                           **kwargs):
        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)

        # extract first two only!
        list_icluster = list_icluster[:2]

        # extract
        (ss_list, [xx, yy, zz]) = self._world.getTube(list_icluster,
                                                      sdwidth, z=z,
                                                      resolution=resolution)

        # to plot
        ss_plot = - np.inf * np.ones_like(ss_list[0])

        # 1 :: blocks added
        ss_added = ((ss_list[0] - ss_list[1])==-1)

        # 2 :: blocks removed
        ss_removed = ((ss_list[0] - ss_list[1])==1)

        # 3 :: present in both
        ss_neutral = ((ss_list[0] + ss_list[1])==2)

        ss_plot[ss_added] = 1.
        ss_plot[ss_removed] = -1.
        ss_plot[ss_neutral] = 0.

        #plt.contourf(xx, yy, ss_plot, levels=[-np.inf, -1., 0., 1., np.inf], colors='none', hatches=['//', '.', '/'], **kwargs)

        plt.contourf(xx,
                     yy,
                     ss_plot,
                     levels=[-np.inf, -1., 0., 1., np.inf],
                     colors=('r','b','g'),
                     alpha=alpha,
                     **kwargs)

        for i in [1, 2, 3]:
            if i == 1:
                ss1 = 1.*ss_removed
                color = 'r'
            elif i == 2:
                ss1 = 1.*ss_added
                color = 'g'
            elif i == 3:
                ss1 = 1.*ss_neutral
                color = 'b'
            # plot an iso surface
            plt.contour(xx, yy, ss1, levels=[0.5], colors=color)

    ## Plot the log-likehood of confidence regions -- which can be related to traffic complexity in the future
    # @param self object pointer
    # @param list_icluster list of clusters to compare
    # @param pmin minimum value on a normalised scale
    # @param pmax maximum value on a normalised scale
    # @param z if specified, it evaluates the confidence region at a constant altitude for 3D trajectories
    # @param resolution specify resolution of region
    def plotLogLikelihood(self,
                          list_icluster=None,
                          pmin=0, pmax=1,
                          z=None,
                          resolution=None):
        # check validity
        list_icluster = self._world._check_list_icluster(list_icluster)




        (ss_list, [xx, yy, zz]) = self._world.getLogLikelihood(list_icluster,
                                                               resolution,
                                                               z)

        ss = ss_list[0] # initialise

        for ss1 in ss_list:
            # find those greater
            mask = np.greater(ss1, ss)
            # replace
            ss[mask] = ss1[mask]

        # normalise
        ss_norm = (ss - np.min(ss)) / (np.max(ss) - np.min(ss))

        # plot contours
        self._ax.pcolor(xx,
                        yy,
                        ss_norm,
                        cmap="viridis",
                        vmin=pmin,
                        vmax=pmax)

    def plotComplexityMap(self,
                          list_icluster=None,
                          complexity=1,
                          pmin=0, pmax=1,
                          z=None,
                          resolution=None, cmap1="Reds"):

        ss, xx, yy, zz = self._world.getComplexityMap(list_icluster,
                                                      complexity,
                                                      resolution,
                                                      z)

        # normalise
        ss_norm = (ss - np.min(ss)) / (np.max(ss) - np.min(ss))

        # plot contours
        cax = self._ax.pcolor(xx,
                              yy,
                              ss_norm,
                              cmap=cmap1,
                              vmin=pmin,
                              vmax=pmax)

        return cax

    ## add colorbar
    def plotColourBar(self, *args, **kwargs):

        cbar = self._fig.colorbar(*args, **kwargs)

        # horizontal colorbar
        # cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])

        return cbar


    ## Plots the title or worldname
    # @param self object pointer
    def _plotTitle(self):
        # add title
        world_name = self._world.getName()
        if not (world_name == None):
            plt.title(world_name)

    ## saves the figure to a file in the output folder
    # @param self object pointer
    # @param add additional identifier for file
    def save(self, add=None):
        if (add==None):
            saveas = self._world.getName()
        else:
            saveas = "{0}_{1}".format(self._world.getName(), add)

        plt.savefig("output/2d_{0}.png".format(saveas))

    ## shows the figure (pop-up or inside notebook)
    # @param self object pointer
    def show(self):
        plt.show()

    ## closes all figures
    # @param self object pointer
    def close(self):
        plt.close("all")
