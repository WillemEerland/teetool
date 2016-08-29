"""
<description>
"""

import numpy as np
import pytest as pt

import teetool as tt

def test_ncolours():
    """
    tests the generation of n distinct colours
    """

    #
    ncolours = "hello World!"
    with pt.raises(TypeError) as testException:
        _ = tt.helpers.getDistinctColours(ncolours)

    #
    ncolours = -1
    with pt.raises(ValueError) as testException:
        _ = tt.helpers.getDistinctColours(ncolours)

    #
    for ncolours in [1,10]:
        colours = tt.helpers.getDistinctColours(ncolours)
        assert (len(colours) == ncolours)

def test_toy_trajectories():
    """
    tests generation of toy trajectories
    """

    ntraj = 50

    for d in [2, 3]:

        traj = tt.helpers.get_trajectories(ntype=0, D=d, N=ntraj)

        assert (len(traj) == ntraj)

        for (x, Y) in traj:
            assert (np.size(Y,1) == d)
