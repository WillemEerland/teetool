# module that produces toy trajectories for testing purposes

"""
-----------------------------------------
- This part generates toy trajectories
-----------------------------------------
"""

import numpy as np

def get_trajectories(ntype=1, D=3, N=50):
    """
    ntype: different output
    d: number of dimensions (2d or 3d)
    returns a list of trajectories (x, Y)
    """
    # remove random effect
    np.random.seed(seed=10)

    # PARAMETERS
    M = 100  # number of data-points per trajectory

    # generate toy trajectories

    x = np.linspace(-50, 50, num=M)

    toy_trajectories = []

    for i in range(N):

        if (ntype == 1):
            # [first set of trajectories]
            y1 = x + 5*np.random.rand(1) - 2.5
            y2 = 0.05*(x**2) + 20*np.random.rand(1) + 80
            y3 = x + 5*np.random.rand(1) - 2.5
        else:
            # [second set of trajectories]
            y1 = x + 5*np.random.rand(1) - 2.5
            y2 = -x + 20*np.random.rand(1) + 50
            y3 = x + 5*np.random.rand(1) - 2.5

        # 2d / 3d
        if (D == 2):
            Y = np.array([y1, y2]).transpose()
        else:
            Y = np.array([y1, y2, y3]).transpose()

        toy_trajectories.append( (x, Y) )

    return toy_trajectories
