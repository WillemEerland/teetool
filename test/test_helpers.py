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
    mcolours = "hello World!"
    with pt.raises(TypeError) as testException:
        _ = tt.helpers.getDistinctColours(mcolours)

    #
    mcolours = -1
    with pt.raises(ValueError) as testException:
        _ = tt.helpers.getDistinctColours(mcolours)

    #
    for mcolours in [1,10]:
        colours = tt.helpers.getDistinctColours(mcolours)
        assert (len(colours) == mcolours)

def test_toy_trajectories():
    """
    tests generation of toy trajectories
    """

    mtraj = 50

    for d in [2, 3]:

        traj = tt.helpers.get_trajectories(0, d, mtraj)

        assert (len(traj) == mtraj)

        for (x, Y) in traj:
            assert (np.size(Y,1) == d)
