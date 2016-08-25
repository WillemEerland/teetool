"""
<description>
"""

import sys

# add library path
sys.path.append('../teetool')

import teetool as tt
import numpy as np
import os
import pytest as pt

def test_initialisation():
	"""
	<description>
	"""

	# test 1
	# set values
	name_1 = "Hello world!"
	D_1 = 2
	world_1 = tt.World(name=name_1, dimension=D_1)
	assert (world_1.name == name_1)
	assert (world_1.D == D_1)

	# test 2
	# default values
	world_2 = tt.World()
	assert (world_2.name == "")
	assert (world_2.D == 3)

	# test 3
	# bad name
	name_3 = 5
	with pt.raises(TypeError) as testException:
		world_3 = tt.World(name_3)

	# bad TYPE dimension
	name_4 = "Hello world!"
	D_4 = "Hello World!"
	with pt.raises(TypeError) as testException:
		world_4 = tt.World(name_4, D_4)

	# bad VALUE dimension
	name_5 = "Hello World!"
	D_5 = 1
	with pt.raises(ValueError) as testException:
		world_5 = tt.World(name_5, D_5)

def test_adding_a_cluster():
	"""
	<description>
	"""

	D = 3 # dimensionality

	# build world
	world_1 = tt.World(name="test", dimension=D)

	# build a correct cluster
	N = 30 # number of trajectories
	M = 50 # number of data-points
	
	x = np.linspace(0,1,M)
	Y = np.zeros((M,D))

	correct_cluster_data = []
	for n in range(N):
		correct_traj_data = (x,Y)
		correct_cluster_data.append(correct_traj_data)

	correct_cluster_name = "correct data"

	# normal operation
	world_1.addCluster(correct_cluster_data, correct_cluster_name)

	# 
	wrong_cluster_name = 5
	with pt.raises(TypeError) as testException:
		world_1.addCluster(correct_cluster_data, wrong_cluster_name)

	# 
	wrong_cluster_data = 5
	with pt.raises(TypeError) as testException:
		world_1.addCluster(wrong_cluster_data, correct_cluster_name)

	# 
	wrong_trajectory_data = 5
	wrong_cluster_data = correct_cluster_data
	wrong_cluster_data.append(wrong_trajectory_data)
	with pt.raises(TypeError) as testException:
		world_1.addCluster(wrong_cluster_data, correct_cluster_name)