"""
<description>
"""

import numpy as np
import pytest as pt
import teetool as tt


def test_visual():
    """
    can produce figures
    """

    # build world
    world_1 = tt.World(name="Example 3D", dimension=3)

    # extreme reduced resolution
    world_1.setResolution(xstep=3, ystep=3, zstep=3)

    # add trajectories
    for ntype in [0]:
        correct_cluster_name = "toy {0}".format(ntype)
        correct_cluster_data = tt.helpers.get_trajectories(ntype, D=3, N=20)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    # test grid
    [xx, yy, zz] = world_1.getGrid()
    assert (xx.shape == yy.shape)

    # model all trajectories
    settings = {}
    settings["model_type"] = "resample"
    settings["mgaus"] = 10

    world_1.buildModel(0, settings)
    world_1.buildLogProbality(0)

    """
    #  this part is Python 2.7 [ONLY] due to Mayavi / VTK dependencies
    for i in [0]:
        # visuals by mayavi
        visual = tt.Visual_3d(world_1, offscreen=True)
        # visualise trajectories
        visual.plotTrajectories([i])
        # visualise intersection
        visual.plotLogProbability([i])
        # visualise outline
        visual.plotOutline()
    """
