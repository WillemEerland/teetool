"""
<description>
"""

import pytest as pt
import teetool as tt


@pt.mark.xfail(reason="out of the blue stopped working on Travis. Mayavi/VTK *sigh*")
def test_visual_3d():
    """
    can produce figures
    """

    # build world
    world_1 = tt.World(name="Example 3D", ndim=3)

    # extreme reduced resolution
    world_1.setResolution(xstep=3, ystep=3, zstep=3)

    # add trajectories
    for mtype in [0, 1]:
        correct_cluster_name = "toy {0}".format(mtype)
        correct_cluster_data = tt.helpers.get_trajectories(mtype,
                                                            ndim=3, ntraj=20)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    # test grid
    [xx, yy, zz] = world_1.getGrid(ndim=3)
    assert (xx.shape == yy.shape)

    list_icluster = [0, 1]

    # model all trajectories
    settings = {}
    settings["model_type"] = "resampling"
    settings["ngaus"] = 10

    world_1.buildModel(list_icluster, settings)
    world_1.buildLogProbality(list_icluster)
    world_1.buildTube(list_icluster)

    #  this part is Python 2.7 [ONLY] due to Mayavi / VTK dependencies

    # visuals by mayavi
    visual = tt.visual_3d.Visual_3d(world_1)
    # enable offscreen rendering
    visual.enableOffScreen()
    # visualise trajectories
    visual.plotTrajectories(list_icluster)
    # visualise intersection
    visual.plotLogProbability(list_icluster)
    # visualise tube
    visual.plotTube(list_icluster)
    # visualise outline
    visual.plotOutline()
    # close
    visual.close()
