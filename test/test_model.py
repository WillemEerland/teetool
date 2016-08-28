"""
<description>
"""

import numpy as np
import pytest as pt

import teetool as tt
from teetool import helpers

def test_init():
    """
    test initialisation of a model with valid data
    """

    ngaus = 100 # number of Gaussian output
    ndim = 3 # number of dimensions

    cluster_data = helpers.get_trajectories(ntype=1, D=ndim)
    valid_settings = {"model_type": "resampling", "mgaus": ngaus}

    # normal operation
    new_model = tt.model.Model(cluster_data, valid_settings)

    """
    *** extend to private functions ***
    """

    # CHECK dimension
    for d in (2,3):
        cluster_data_d = helpers.get_trajectories(ntype=1, D=d, N=10)
        assert (new_model._getDimension(cluster_data_d) == d)

    # normalise data
    cluster_data = helpers.get_trajectories(ntype=1, D=ndim)
    norm_cluster_data = new_model._normalise_data(cluster_data)

    # CHECK if trajectories are normalised
    (xmin, xmax) = new_model._getMinMax(norm_cluster_data)
    assert (xmin == 0)
    assert (xmax == 1)

    # model by resampling
    (mu_y, sig_y) = new_model._model_by_resampling(norm_cluster_data, ngaus)

    # CHECK dimensions
    assert (mu_y.shape == ((ndim*ngaus),1)) # [ ndim*ngaus x 1 ]
    assert (sig_y.shape == (ndim*ngaus,ndim*ngaus)) # [ ndim*ngaus x ndim*ngaus ]

    # convert to cells
    (cc, cA) = new_model._getGMMCells(mu_y, sig_y)

    # CHECK numbers
    assert (len(cc) == ngaus)
    assert (len(cA) == ngaus)

    # CHECK dimensions
    assert (cc[0].shape == (ndim,1)) #  [ndim x 1]
    assert (cA[0].shape == (ndim,ndim)) #  [ndim x ndim]

def test_eval():
    """
    testing the evaluation of a initialised model
    """

    ngaus = 100 # number of Gaussian output

    # 3d
    ndim = 3 # number of dimensions

    cluster_data = helpers.get_trajectories(ntype=1, D=ndim)
    valid_settings = {"model_type": "resampling", "mgaus": ngaus}

    # normal operation
    new_model_3d = tt.model.Model(cluster_data, valid_settings)

    # 2d
    ndim = 2 # number of dimensions

    cluster_data = helpers.get_trajectories(ntype=1, D=ndim)
    valid_settings = {"model_type": "resampling", "mgaus": ngaus}

    # normal operation
    new_model_2d = tt.model.Model(cluster_data, valid_settings)

    """
    test function
    """

    # grid to test
    x, y, z = np.ogrid[-60:60:3j, -10:240:3j, -60:60:3j]

    s = new_model_3d.eval(x, y, z)

    # CHECK dimension
    nx = np.size(x, 0)
    ny = np.size(y, 1)
    nz = np.size(z, 2)

    assert (s.shape == (nx, ny, nz))

    """
    test subfunctions
    """

    # _gauss_logLc

    y = np.mat([[1],[1]])

    pL = new_model_2d._gauss_logLc(y)

    assert (pL.shape == (1,1))

    # _gauss
    y = np.mat([[0],[1]]) # column vector
    c = np.mat([[0],[0]]) # column vector centre
    A = np.mat([[1,0],[0,1]]) # matrix covariance

    pL = new_model_2d._gauss(y, c, A)

    assert (pL.shape == (1,1))
