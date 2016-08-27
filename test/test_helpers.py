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
