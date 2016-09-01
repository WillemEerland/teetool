"""
<description>
"""

import numpy as np
import pytest as pt

import teetool as tt

def test_basis():
    """
    testing basis class
    """

    cluster_data = tt.helpers.get_trajectories(ntype=1, ndim=2)
    valid_settings = {"model_type": "resampling", "ngaus": 10}
    model_1 = tt.model.Model(cluster_data, valid_settings)

    mpoints = 10
    x_test = np.linspace(0, 1, mpoints)

    mbasis = 5
    mdim = 3

    # test exception
    with pt.raises(NotImplementedError) as testException:
        _ = model_1._Basis(basisType="Hello World!", nbasis=mbasis, ndim=mdim)

    myBasis = model_1._Basis(basisType="rbf", nbasis=mbasis, ndim=mdim)

    # test Gaussian
    res = myBasis._funcRbf(x=0, mu1=0, sig1=1)
    assert(np.isfinite(res))

    res = myBasis._getBasisRbfVector(x_sca=0, nbasis=mbasis)
    assert(res.shape == (1, mbasis))
    #
    assert (np.any(np.isfinite(res)))

    # test bernstein
    res = myBasis._getBasisBernsteinVector(x_sca=0, nbasis=mbasis)
    assert(res.shape == (1, mbasis))
    #
    assert (np.any(np.isfinite(res)))

    # test assortment
    for mbasis in [5]:
        for mdim in [2]:
            for mtype in ["rbf", "bernstein"]:
                # settings
                myBasis = model_1._Basis(basisType=mtype, nbasis=mbasis, ndim=mdim)
                # obtain basis
                H = myBasis.get(x_test)
                # shape [mpoints x mgaus]
                assert (H.shape == (mpoints*mdim, mbasis*mdim))
                # all finite numbers
                assert (np.any(np.isfinite(res)))
