"""
<description>
"""

import numpy as np
import pytest as pt

import teetool as tt


def test_inside():
    """
    tests if points are inside
    """

    mdim = 2

    cluster_data = tt.helpers.get_trajectories(1, mdim)
    valid_settings = {"model_type": "resampling", "ngaus": 100}

    # normal operation
    new_model = tt.model.Model(cluster_data, valid_settings)

    #
    (x, Y) = cluster_data[0]

    # insane width..
    p_test = new_model.isInside_pnts(Y, sdwidth=30)

    assert(p_test.all())

    # 2d
    [xx, yy] = np.mgrid[-10:10:2j, -10:10:2j]

    ss = new_model.isInside_grid(sdwidth=1, xx=xx, yy=yy)

    # test points2grid
    (Y_pos, Y_idx) = new_model._grid2points(xx, yy)

    # evaluate some points
    s = new_model.isInside_pnts(Y_pos)

    # convert back to grid
    ss2 = new_model._points2grid(s, Y_idx)

    # should be identical
    np.testing.assert_array_almost_equal_nulp(ss, ss2)


def test_help_func():
    """
    testing various functions
    """

    for mdim in [2, 3]:

        mgaus = 10  # number of Gaussian output
        # mdim = 3  # number of dimensions

        cluster_data = tt.helpers.get_trajectories(1, mdim)
        valid_settings = {"model_type": "resampling", "ngaus": mgaus}

        # normal operation
        new_model = tt.model.Model(cluster_data, valid_settings)

        # generating samples

        if (mdim == 2):
            c = np.mat([[0], [0]])
            A = np.mat([[1,0],[0,1]])
        else:
            c = np.mat([[0], [0], [0]])
            A = np.mat([[1,0,0],[0,1,0],[0,0,1]])

        Y = new_model._getSample(c, A, 1)
        assert(Y.shape == (1, mdim))

        Y = new_model._getSample(c, A, 3)
        assert(Y.shape == (3, mdim))

        # generating coordinates
        msamples = 10
        Y = new_model._getCoordsEllipse(nsamples=msamples)

        #assert(Y.shape == (mgaus*msamples, mdim))

        # all finite
        assert(np.isfinite(Y).all())


def test_init():
    """
    test initialisation of a model with valid data
    """

    mgaus = 10  # number of Gaussian output
    mdim = 3  # number of dimensions

    cluster_data = tt.helpers.get_trajectories(1, mdim)
    valid_settings = {"model_type": "resampling", "ngaus": mgaus}

    # normal operation
    new_model = tt.model.Model(cluster_data, valid_settings)

    """
    *** extend to private functions ***
    """

    # CHECK dimension
    for mdim2 in (2, 3):
        cluster_data_d = tt.helpers.get_trajectories(ntype=1,
                                                     ndim=mdim2, ntraj=10)
        assert (tt.helpers.getDimension(cluster_data_d) == mdim2)

    # normalise data
    cluster_data = tt.helpers.get_trajectories(1, mdim)
    norm_cluster_data = tt.helpers.normalise_data(cluster_data)

    # CHECK if trajectories are normalised
    (xmin, xmax) = tt.helpers.getMinMax(norm_cluster_data)
    assert (xmin == 0)
    assert (xmax == 1)

    # model by resampling
    gp = tt.gaussianprocess.GaussianProcess(norm_cluster_data, mgaus)

    (mu_y, sig_y, cc, cA) = gp.model_by_resampling()

    # CHECK dimensions
    assert (mu_y.shape == ((mdim*mgaus), 1))  # [ mdim*mgaus x 1 ]
    # [ mdim*mgaus x mdim*mgaus ]
    assert (sig_y.shape == (mdim*mgaus, mdim*mgaus))

    # convert to cells
    (cc, cA) = gp._getGMMCells(mu_y, sig_y, mgaus)

    # CHECK numbers
    assert (len(cc) == mgaus)
    assert (len(cA) == mgaus)

    # CHECK dimensions
    assert (cc[0].shape == (mdim, 1))  # [mdim x 1]
    assert (cA[0].shape == (mdim, mdim))  # [mdim x mdim]

    # CHECK
    with pt.raises(ValueError) as testException:
        _, yy = np.mgrid[-10:10:2j, -10:10:2j]
        xx, _, _ = np.mgrid[-10:10:2j, -10:10:2j, -10:10:2j]
        _ = new_model.evalLogLikelihood(xx, yy)
