import numpy as np
import pytest as pt

import teetool as tt

def test_subfunctions_EM():
    """test subfunctions that contribute to the EM algorithm
    """

    cluster_data = tt.helpers.get_trajectories(ntype=1, ndim=2)

    # TEMPORARY FAKE
    valid_settings = {"model_type": "EM",
                      "ngaus": 10,
                      "nbasis": 5,
                      "basis_type": "rbf"}

    # normal operation
    gp = tt.gaussianprocess.GaussianProcess(cluster_data, ngaus=10)

    # create a basis (normally done inside model)
    basis = tt.basis.Basis("rbf", nbasis=5, ndim=2)

    # simple conversion
    (yc, Hc) = gp._from_clusterdata2cells(cluster_data, basis)

    assert(len(yc) == len(Hc))

    # average
    Ewc = []
    Ewc.append(np.array([0.0, 0.0, 0.0]))
    Ewc.append(np.array([2.0, 2.0, 2.0]))

    mu_w = gp._E_mu(Ewc)
    mu_w_expected = np.array([1.0, 1.0, 1.0])

    np.testing.assert_array_almost_equal_nulp(mu_w, mu_w_expected)

    # sigma (loose check)
    # sig_w = model._E_sigma(mu_w, )
