"""
<description>
"""

import pytest as pt
import teetool as tt


def test_visual_2d():
    """
    produce figures
    """

    mdim = 2
    mtraj = 20

    # build world
    world_1 = tt.World(name="Example 3D", ndim=2, nres=3)

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

    world_1.buildModel(list_iclusters, settings)

    # visuals by
    visual = tt.visual_2d.Visual_2d(world_1)
    # visualise trajectories
    visual.plotTrajectories(list_iclusters)
    # visualise intersection
    visual.plotLogLikelihood()
    # close
    visual.close()
