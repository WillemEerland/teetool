"""
<description>
"""

import pytest as pt
pt.importorskip("teetool.visual_2d")
import teetool as tt


def test_visual_2d():
    """
    produce figures
    """

    # from teetool import visual_2d

    mdim = 2
    mtraj = 20

    # build world
    world_1 = tt.World(name="Example 3D", ndim=2)

    # extreme reduced resolution
    world_1.setResolution(xstep=3, ystep=3, zstep=3)

    # add trajectories
    for mtype in [0, 1]:
        correct_cluster_name = "toy {0}".format(mtype)
        correct_cluster_data = tt.helpers.get_trajectories(mtype, ndim=mdim, ntraj=mtraj)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    # test grid
    [xx, yy] = world_1.getGrid()
    assert (xx.shape == yy.shape)

    # model all trajectories
    settings = {}
    settings["model_type"] = "resampling"
    settings["ngaus"] = 10

    for i in [0, 1]:
        world_1.buildModel(i, settings)
        world_1.buildLogProbality(i)

    for i in [0, 1]:
        # visuals by
        visual = tt.visual_2d.Visual_2d(world_1)
        # visualise trajectories
        visual.plotTrajectories([i])
        # visualise intersection
        visual.plotLogProbability([i])
        # close
        visual.close()

    # visuals by
    visual = tt.visual_2d.Visual_2d(world_1)
    # visualise trajectories
    visual.plotTrajectories([0, 1])
    # visualise intersection
    visual.plotLogProbability([0, 1])
    # close
    visual.close()

    assert (visual.plotOutline() == True)
