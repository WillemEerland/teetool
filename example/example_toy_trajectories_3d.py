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

# print overview
new_world.overview()

# model all trajectories

settings = {}
settings["model_type"] = "resample"
settings["mgaus"] = 50

new_world.model(settings)

# print overview
new_world.overview()

"""
this part is Python 2.7 [ONLY] due to Mayavi / VTK dependencies
"""

# visualise trajectories using mayavi
#visual_3d.show_trajectories(new_world)

x, y, z = np.ogrid[-60:60:30j, -10:240:30j, -60:60:30j]

visual_3d.show_intersection(new_world, x, y, z)
