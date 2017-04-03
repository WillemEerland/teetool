"""
<description>
"""

import pytest as pt
import teetool as tt


#  @pt.mark.xfail(reason="out of the blue stopped working on Travis. Mayavi/VTK *sigh*")
def test_visual_3d():
    """
    can produce figures
    """

    # build world
    world_1 = tt.World(name="Example 3D", ndim=3, resolution=[10, 10, 10])

    # extreme reduced resolution
    # world_1.setResolution(xstep=3, ystep=3, zstep=3)

    # add trajectories
    for mtype in [0, 1]:
        correct_cluster_name = "toy {0}".format(mtype)
        correct_cluster_data = tt.helpers.get_trajectories(mtype,
                                                            ndim=3, ntraj=20)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    list_icluster = [0, 1]

    # model all trajectories
    settings = {}
    settings["model_type"] = "resampling"
    settings["ngaus"] = 10

    world_1.buildModel(settings)

    #  this part is Python 2.7 [ONLY] due to Mayavi / VTK dependencies

    # visuals by mayavi
    visual = tt.visual_3d.Visual_3d(world_1)
    # visualise trajectories
    visual.plotTrajectories(list_icluster, ntraj=3)
    # visualise points
    # visualise trajectories
    visual.plotTrajectoriesPoints(x1=0.0,list_icluster=None, ntraj=3)
    # visualise intersection
    visual.plotLogLikelihood()
    #
    visual.plotComplexityMap()
    # visualise tube
    visual.plotTube()
    # visualise difference
    visual.plotTubeDifference([0, 1])
    # visualise outline
    visual.plotOutline()
    # close
    visual.close()
