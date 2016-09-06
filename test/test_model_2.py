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
