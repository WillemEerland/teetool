"""
<description>
"""

import numpy as np
import pytest as pt
import sys

import teetool as tt

if not ((sys.version_info[0] == 2) and (sys.version_info[1] == 7)):
    # Mayavi not availabe

    def test_no_import():
        """
        tests that Mayavi is not installed - otherwise we could use it! :)
        """
        with pt.raises(ImportError) as testException:
            from teetool import visual_3d
else:
    # Mayavi should be installed for Python version 2.7

    def test_yes_import():
        """
        should import when installed
        """
        # Python 2.7
        from teetool import visual_3d

    def test_visual():
        """
        can produce figures
        """

        from teetool import visual_3d

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

        #  this part is Python 2.7 [ONLY] due to Mayavi / VTK dependencies
        for i in [0]:
            # visuals by mayavi
            visual = visual_3d.Visual_3d(world_1)
            # visualise trajectories
            visual.plotTrajectories([i])
            # visualise intersection
            visual.plotLogProbability([i])
            # visualise outline
            visual.plotOutline()
