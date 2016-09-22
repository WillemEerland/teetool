import numpy as np
import pytest as pt

import teetool as tt

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

        # tube
        if (mdim == 2):
            xx, yy = np.mgrid[-10:10:2j, -10:10:2j]
            ss = new_model.isInside_grid(sdwidth=1, xx=xx, yy=yy)
            assert (ss.shape==xx.shape)
        if (mdim == 3):
            xx, yy, zz = np.mgrid[-10:10:2j, -10:10:2j, -10:10:2j]
            ss = new_model.isInside_grid(sdwidth=1, xx=xx, yy=yy, zz=zz)
            assert (ss.shape==xx.shape)

        # log likelihood
        if (mdim == 2):
            xx, yy = np.mgrid[-10:10:2j, -10:10:2j]
            ss = new_model.evalLogLikelihood(xx, yy)
            assert (ss.shape==xx.shape)
        if (mdim == 3):
            xx, yy, zz = np.mgrid[-10:10:2j, -10:10:2j, -10:10:2j]
            ss = new_model.evalLogLikelihood(xx, yy, zz)
            assert (ss.shape==xx.shape)

        # test subfunctions

        y = np.zeros((mdim, 1))
        y = np.mat(y)

        pL = tt.helpers.gauss_logLc(y, mdim, new_model._cc, new_model._cA)

        assert (np.isfinite(pL))

        c = np.zeros((mdim, 1))
        A = np.eye(mdim)

        pL = tt.helpers.gauss(y, mdim, c, A)

        assert (np.isfinite(pL))

    #
    for d in [2, 3]:
        do_this_test(mdim=d, model_type="resampling")

    for model_type1 in ["ML", "EM"]:
        for basis_type1 in ["rbf", "bernstein"]:
            do_this_test(mdim=2, model_type=model_type1, basis_type=basis_type1)
