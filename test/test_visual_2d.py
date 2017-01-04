"""
<description>
"""

import pytest as pt
import numpy as np
import teetool as tt


def test_visual_2d():
    """
    produce figures
    """

    mdim = 2
    mtraj = 20

    # build world
    world_1 = tt.World(name="Example 3D", ndim=2, resolution=100)

    # extreme reduced resolution
    #world_1.setResolution(xstep=3, ystep=3, zstep=3)

    # add trajectories
    for mtype in [0, 1]:
        correct_cluster_name = "toy {0}".format(mtype)
        correct_cluster_data = tt.helpers.get_trajectories(mtype, ndim=mdim, ntraj=mtraj)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    list_iclusters = [0, 1]

    # model all trajectories
    settings = {}
    settings["model_type"] = "resampling"
    settings["ngaus"] = 10

    world_1.buildModel(settings)

    # visuals by
    visual = tt.visual_2d.Visual_2d(world_1)
    # visualise intersection
    visual.plotLogLikelihood()
    # close
    visual.close()

    # visuals by
    visual = tt.visual_2d.Visual_2d(world_1)
    # visualise intersection
    visual.plotTube()
    # close
    visual.close()

    # visuals by
    visual = tt.visual_2d.Visual_2d(world_1)
    # visualise intersection
    visual.plotTubeDifference()
    # close
    visual.close()

    # new figure
    visual = tt.visual_2d.Visual_2d(world_1)
    # visualise trajectories
    visual.plotTrajectories(list_iclusters)
    # visualise points
    visual.plotTrajectoriesPoints(x1=0.0, list_icluster=None)
    # mean
    visual.plotMean()
    #
    visual.plotSamples()

    #
    visual.plotBox(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

    # close
    visual.close()

def test_subplots():
    """test time-series
    """

    mdim = 2
    mtraj = 20

    # build world
    world_1 = tt.World(name="Example 3D", ndim=2, resolution=100)

    # extreme reduced resolution
    #world_1.setResolution(xstep=3, ystep=3, zstep=3)

    # add trajectories
    for mtype in [0, 1]:
        correct_cluster_name = "toy {0}".format(mtype)
        correct_cluster_data = tt.helpers.get_trajectories(mtype, ndim=mdim, ntraj=mtraj)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    list_iclusters = [0, 1]

    # model all trajectories
    settings = {}
    settings["model_type"] = "resampling"
    settings["ngaus"] = 10

    world_1.buildModel(settings)

    # visuals by
    visual = tt.visual_2d.Visual_2d(world_1)

    visual.plotTimeSeries()

    visual.close()
