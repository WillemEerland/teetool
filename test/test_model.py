"""
<description>
"""

import numpy as np
import pytest as pt

import teetool as tt

def test_init():
    """
    test initialisation of a model with valid data
    """

    ngaus = 100  # number of Gaussian output
    ndim = 3  # number of dimensions

    cluster_data = tt.helpers.get_trajectories(ntype=1, D=ndim)
    valid_settings = {"model_type": "resampling", "mgaus": ngaus}

    # normal operation
    new_model = tt.model.Model(cluster_data, valid_settings)

    """
    *** extend to private functions ***
    """

    # CHECK dimension
    for d in (2, 3):
        cluster_data_d = tt.helpers.get_trajectories(ntype=1, D=d, N=10)
        assert (new_model._getDimension(cluster_data_d) == d)

    # normalise data
    cluster_data = tt.helpers.get_trajectories(ntype=1, D=ndim)
    norm_cluster_data = new_model._normalise_data(cluster_data)

    # CHECK if trajectories are normalised
    (xmin, xmax) = new_model._getMinMax(norm_cluster_data)
    assert (xmin == 0)
    assert (xmax == 1)

    # model by resampling
    (mu_y, sig_y) = new_model._model_by_resampling(norm_cluster_data, ngaus)

    # CHECK dimensions
    assert (mu_y.shape == ((ndim*ngaus), 1))  # [ ndim*ngaus x 1 ]
    # [ ndim*ngaus x ndim*ngaus ]
    assert (sig_y.shape == (ndim*ngaus, ndim*ngaus))

    # convert to cells
    (cc, cA) = new_model._getGMMCells(mu_y, sig_y)

    # CHECK numbers
    assert (len(cc) == ngaus)
    assert (len(cA) == ngaus)

    # CHECK dimensions
    assert (cc[0].shape == (ndim, 1))  # [ndim x 1]
    assert (cA[0].shape == (ndim, ndim))  # [ndim x ndim]


def test_eval():
    """
    testing the evaluation of a initialised model
    """

    ngaus = 100  # number of Gaussian output

    # 2d / 3d
    for ndim in (2, 3):
        cluster_data = tt.helpers.get_trajectories(ntype=1, D=ndim)
        valid_settings = {"model_type": "resampling", "mgaus": ngaus}
        # normal operation
        new_model = tt.model.Model(cluster_data, valid_settings)

        if (ndim == 2):
            xx, yy = np.mgrid[-10:10:3j, -10:10:3j]
            s = new_model.eval(xx, yy)
        if (ndim == 3):
            xx, yy, zz = np.mgrid[-10:10:3j, -10:10:3j, -10:10:3j]
            s = new_model.eval(xx, yy, zz)

        assert(s.shape == xx.shape)

        # test subfunctions

        y = np.zeros((ndim, 1))
        y = np.mat(y)

        pL = new_model._gauss_logLc(y)

        assert (pL.shape == (1, 1))

        c = np.zeros((ndim, 1))
        A = np.eye(ndim)

        pL = new_model._gauss(y, c, A)

        assert (pL.shape == (1, 1))
