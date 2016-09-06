"""
<example>
"""

import teetool as tt  # core
from teetool import visual_3d


llsettings = []

llsettings.append(["resampling", 100, "", "", 0])
#llsettings.append(["ML", 100, "bernstein", 5, 0])
#llsettings.append(["ML", 100, "rbf", 10, 0])

#llsettings.append(["resampling", 100, "", "", .5])
#llsettings.append(["ML", 100, "bernstein", 5, .5])
#llsettings.append(["ML", 100, "rbf", 10, .5])

#llsettings.append(["EM", 100, "bernstein", 5, .5])
#llsettings.append(["EM", 100, "rbf", 10, .5])

for ls in llsettings:

    settings = {"model_type":ls[0],
                "ngaus":ls[1],
                "basis_type":ls[2],
                "nbasis":ls[3]}

    # build world
    world_name = "[{0}] [{1}] [{2}] [{3}]".format(settings["model_type"],
                                  settings["basis_type"],
                                  settings["nbasis"],
                                  ls[4])

    # build world
    new_world = tt.World(name=world_name, ndim=3)

    # modify default resolution
    new_world.setResolution(xstep=25, ystep=25, zstep=15)

    # add trajectories
    for ntype in [0, 1]:
        cluster_name = "toy {0}".format(ntype)
        cluster_data = tt.helpers.get_trajectories(ntype,
                                                   ndim=3,
                                                   ntraj=50,
                                                   npoints=100,
                                                   noise_std=ls[4])
        new_world.addCluster(cluster_data, cluster_name)

    # overview
    new_world.overview()

    # model
    new_world.buildModel(0, settings)
    new_world.overview()  # overview
    new_world.buildModel(1, settings)
    new_world.overview()  # overview

    # log
    new_world.buildLogProbality(0)
    new_world.overview()  # overview
    new_world.buildLogProbality(1)
    new_world.overview()  # overview

    #  this part is Python 2.7 [ONLY] due to Mayavi / VTK dependencies

    for i in [0, 1]:
        # visuals by mayavi
        visual = visual_3d.Visual_3d(new_world)
        # visualise trajectories
        visual.plotTrajectories([i])
        # visualise intersection
        visual.plotLogProbability([i])
        # visualise outline
        visual.plotOutline()
        # save
        # visual.save()


    # visuals by mayavi
    visual = visual_3d.Visual_3d(new_world)
    # visualise trajectories
    visual.plotTrajectories([0, 1])
    # visualise intersection
    visual.plotLogProbability([0, 1])
    # visualise outline
    visual.plotOutline()
    # save
    #visual.save()

    # show [ requires user input ]
    visual.show()
