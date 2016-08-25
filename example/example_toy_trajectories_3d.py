
import numpy as np

# remove random effect
np.random.seed(seed=10)

# PARAMETERS
N = 50 # number of trajectories
M = 100 # number of data-points per trajectory
D = 3 # number of dimensions

# generate toy trajectories

x = np.linspace(-50,50,num=M)

# [first set of trajectories] -------------------------------------------

toy_trajectories_1 = []

for i in range(N):
       
    y1 = x + 5*np.random.rand(1) - 2.5
    y2 = 0.05*(x**2) + 20*np.random.rand(1) + 80
    y3 = x + 5*np.random.rand(1) - 2.5
    
    Y = np.array([y1,y2,y3]).transpose()
    
    this_trajectory = (x, Y)

    toy_trajectories_1.append(this_trajectory)

# [second set of trajectories] ----------------------------------------------

toy_trajectories_2 = []

for i in range(N):
    
    y1 = x + 5*np.random.rand(1) - 2.5
    y2 = -x + 20*np.random.rand(1) + 50
    y3 = x + 5*np.random.rand(1) - 2.5

    Y = np.array([y1,y2,y3]).transpose()

    this_trajectory = (x, Y)

    toy_trajectories_2.append(this_trajectory)

# (optional) visualise trajectories using mayavi

import mayavi.mlab as mlab

black = (0,0,0)
white = (1,1,1)
red = (1,0,0)
blue = (0,0,1)

mlab.figure()

for this_trajectory in toy_trajectories_1:
    (x, Y) = this_trajectory
    mlab.plot3d(Y[:,0], Y[:,1], Y[:,2], color=red, tube_radius=.1)

for this_trajectory in toy_trajectories_2:
    (x, Y) = this_trajectory
    mlab.plot3d(Y[:,0], Y[:,1], Y[:,2], color=blue, tube_radius=.1)

mlab.show()

# add trajectories to world

