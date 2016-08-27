"""
<example>
"""

import numpy as np
import sys

import teetool as tt
import test

# parameters
ntraj = 50
ndim = 3

# build world
new_world = tt.World(name="Example 3D", dimension=ndim)

# add trajectories
for ntype in [1,2]:
    cluster_name = "toy {}".format(ntype)
    cluster_data = test.toy.get_trajectories(ntype, D=ndim, N=ntraj)
    new_world.addCluster(cluster_data, cluster_name)

# print overview
new_world.overview()

# model all trajectories
new_world.model()

# print overview
new_world.overview()

# visualise trajectories using mayavi
new_world.show_trajectories()

new_world.show_model()
