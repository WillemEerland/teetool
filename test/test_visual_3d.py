"""
<description>
"""

import pytest as pt
import teetool as tt
pt.importorskip("teetool.visual_3d")


def test_visual_3d():
    """
    can produce figures
    """

    from teetool import visual_3d

    # build world
    world_1 = tt.World(name="Example 3D", dimension=3)

    # extreme reduced resolution
    world_1.setResolution(xstep=3, ystep=3, zstep=3)

    # add trajectories
    for ntype in [0, 1]:
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

    for i in [0, 1]:
        world_1.buildModel(i, settings)
        world_1.buildLogProbality(i)

    #  this part is Python 2.7 [ONLY] due to Mayavi / VTK dependencies
    for i in [0, 1]:
        # visuals by mayavi
        visual = visual_3d.Visual_3d(world_1)
        # visualise trajectories
        visual.plotTrajectories([i])
        # visualise intersection
        visual.plotLogProbability([i])
        # visualise outline
        visual.plotOutline()
        # close
        visual.close()

    # visuals by mayavi
    visual = visual_3d.Visual_3d(world_1)
    # visualise trajectories
    visual.plotTrajectories([0, 1])
    # visualise intersection
    visual.plotLogProbability([0, 1])
    # visualise outline
    visual.plotOutline()
    # close
    visual.close()
