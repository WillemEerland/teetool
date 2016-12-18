import numpy as np
import pytest as pt

import teetool as tt

def test_ks():
    """test ks statistic properties"""



    # obtain cluster_data
    cluster_data = tt.helpers.get_trajectories(ntype=0,
                                               ndim=2,
                                               noise_std=.01,
                                               ntraj=10,
                                               npoints=100)

    # create a model
    valid_settings = {"model_type": "resampling", "ngaus": 100}
    model = tt.model.Model(cluster_data, valid_settings)

    # from clusterdata to points
    Y = model._clusterdata2points(cluster_data)

    # (number of trajectories x number of points) x dimension
    assert (Y.shape == (10*100, 2))

    sampled_data = model.getSamples(20)

    S = model._clusterdata2points(sampled_data)

    # (numbers of samples x resolution model) x dimension
    assert (S.shape == (20*100, 2))

    sigma_arr = np.array([1.0, 2.0])

    ks1, lY, lS, xx = model.getKS(cluster_data,
                                  nsamples=10,
                                  sigma_arr=sigma_arr)

    assert (ks1 >= 0)

    # should provide a large mismatch
    cluster_data_wrong = tt.helpers.get_trajectories(ntype=1,
                                                     ndim=2,
                                                     noise_std=.01,
                                                     ntraj=10,
                                                     npoints=100)

    ks2, lY, lS, xx = model.getKS(cluster_data_wrong,
                                  nsamples=10,
                                  sigma_arr=sigma_arr)

    # this tests if the first set is more like the model than the second where the first set is the data put into the model and the second set is completely different
    assert (ks1 < ks2)
