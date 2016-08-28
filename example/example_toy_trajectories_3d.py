"""
<example>
"""

import numpy as np

import teetool as tt  # core
from teetool import helpers  # generate sample trajectories
from teetool import visual_3d  # plot results in Mayavi

# parameters
ntraj = 50
ndim = 3

# build world
new_world = tt.World(name="Example 3D", dimension=ndim)

# add trajectories
for ntype in [1,2]:
    cluster_name = "toy {0}".format(ntype)
    cluster_data = helpers.get_trajectories(ntype, D=ndim, N=ntraj)
    new_world.addCluster(cluster_data, cluster_name)

# model all trajectories

settings = {}
settings["model_type"] = "resample"
settings["mgaus"] = 100

new_world.model(settings)

# print overview
new_world.overview()

"""
this part is Python 2.7 [ONLY] due to Mayavi / VTK dependencies
"""

visual = tt.Visual_3d()

# visualise trajectories using mayavi
visual.add_trajectories(new_world)

# visualise intersection (runs simulation)
#x, y, z = np.mgrid[-60:60:20j, -10:240:40j, -60:60:20j]
#visual.add_intersection(new_world, x, y, z)

# show
visual.show()
