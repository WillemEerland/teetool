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
        assert (new_model._getDimension(cluster_data_d) == mdim2)

    # normalise data
    cluster_data = tt.helpers.get_trajectories(1, mdim)
    norm_cluster_data = new_model._normalise_data(cluster_data)

    # CHECK if trajectories are normalised
    (xmin, xmax) = new_model._getMinMax(norm_cluster_data)
    assert (xmin == 0)
    assert (xmax == 1)

    # model by resampling
    (mu_y, sig_y) = new_model._model_by_resampling(norm_cluster_data, mgaus)

    # CHECK dimensions
    assert (mu_y.shape == ((mdim*mgaus), 1))  # [ mdim*mgaus x 1 ]
    # [ mdim*mgaus x mdim*mgaus ]
    assert (sig_y.shape == (mdim*mgaus, mdim*mgaus))

    # convert to cells
    (cc, cA) = new_model._getGMMCells(mu_y, sig_y, mgaus)

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
        _ = new_model.eval(xx, yy)

def test_eval():
    """
    testing the evaluation of a initialised model
    """

    def do_this_test(mdim, model_type, basis_type=None):

        cluster_data = tt.helpers.get_trajectories(1, mdim)

        valid_settings = {"model_type": model_type, "ngaus": 10}

        if not (basis_type is None):
            valid_settings["basis_type"] = basis_type
            valid_settings["nbasis"] = 5

        # normal operation
        new_model = tt.model.Model(cluster_data, valid_settings)

        if (mdim == 2):
            xx, yy2 = np.mgrid[-10:10:2j, -10:10:2j]
            s = new_model.eval(xx, yy2)
        if (mdim == 3):
            xx, yy, zz = np.mgrid[-10:10:2j, -10:10:2j, -10:10:2j]
            s = new_model.eval(xx, yy, zz)

        assert(s.shape == xx.shape)

        # test subfunctions

        y = np.zeros((mdim, 1))
        y = np.mat(y)

        pL = new_model._gauss_logLc(y)

        assert (pL.shape == (1, 1))

        c = np.zeros((mdim, 1))
        A = np.eye(mdim)

        pL = new_model._gauss(y, c, A)

        assert (pL.shape == (1, 1))

    #
    for d in [2, 3]:
        do_this_test(mdim=d, model_type="resampling")

    for model_type1 in ["ML", "EM"]:
        for basis_type1 in ["rbf", "bernstein"]:
            do_this_test(mdim=2, model_type=model_type1, basis_type=basis_type1)
