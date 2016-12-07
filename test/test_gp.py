import teetool as tt
import numpy as np
import pytest as pt


def produce_cluster_data():
    """returns some sample cluster data with known properties

    output:
        cluster_data - 5 trajectories, constant y (2nd dim), moving from left to right (1st dim)
    """

    # [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
    covariate = np.linspace(0, 1, 11)
    # [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9  1. ]
    horizontal = np.linspace(0, 1, 11)
    # [-1.  -0.5  0.   0.5  1. ]
    vals = np.linspace(-1, 1, 5)

    # recall, cluster_data is a list of (x, Y)
    cluster_data = []

    for val in vals:
        vertical = np.ones_like(horizontal) * val * 1.0

        # shape [11, 2]
        Y = np.array([horizontal, vertical]).transpose()

        cluster_data.append((covariate, Y))

    return cluster_data

def test_cluster_data():
    """check if cluster_data is as expected"""

    cluster_data = produce_cluster_data()

    assert( len(cluster_data) == 5 )

    for (x, Y) in cluster_data:

        assert( x.shape == (11,) )

        assert( Y.shape == (11, 2) )
