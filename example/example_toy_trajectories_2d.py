"""
<example>
"""

import teetool as tt  # core
from teetool import visual_2d

lsettings = []

"""

settings = {"model_type":"resampling",
            "ngaus":100,
            "basis_type":"",
            "nbasis":""}

lsettings.append(settings)

settings = {"model_type":"ML",
            "ngaus":100,
            "basis_type":"bernstein",
            "nbasis":5}

lsettings.append(settings)


settings = {"model_type":"EM",
            "ngaus":100,
            "basis_type":"bernstein",
            "nbasis":5}

lsettings.append(settings)

"""

settings = {"model_type":"ML",
            "ngaus":100,
            "basis_type":"rbf",
            "nbasis":20}

lsettings.append(settings)

settings = {"model_type":"EM",
            "ngaus":100,
            "basis_type":"rbf",
            "nbasis":20}

lsettings.append(settings)


for settings in lsettings:

    # build world
    world_name = "[{0}] [{1}] [{2}]".format(settings["model_type"],
                                  settings["basis_type"],
                                  settings["nbasis"])

    # create a new world
    new_world = tt.World(name=world_name, ndim=2)

    # add trajectories
    for ntype in [0, 1]:
        cluster_name = "toy {0}".format(ntype)
        cluster_data = tt.helpers.get_trajectories(ntype, ndim=2, ntraj=50)
        new_world.addCluster(cluster_data, cluster_name)

    # output an overview
    new_world.overview()

    # build the model
    new_world.buildModel(0, settings)
    new_world.buildModel(1, settings)

    # modify default resolution
    new_world.setResolution(xstep=25, ystep=25)

    # build the log-probability for the set grid (resolution)
    new_world.buildLogProbality(0)
    new_world.buildLogProbality(1)

    # output an overview
    new_world.overview()

    # visuals by mayavi
    visual = visual_2d.Visual_2d(new_world)
    # visualise trajectories
    visual.plotTrajectories([0, 1])
    # visualise samples
    visual.plotSamples([0, 1])
    # legend
    visual.plotLegend()
    # visualise intersection
    visual.plotLogProbability([0, 1])

# show [ wait for user input ]
visual.show()
