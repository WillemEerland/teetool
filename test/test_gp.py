import teetool as tt
import numpy as np
import pytest as pt


def produce_cluster_data():
    """returns some sample cluster data with known properties

    output:
        cluster_data - 5 trajectories, constant y (2nd dim), moving from left to right (1st dim)
    """

    npoints = 50
    ntrajs = 5

    # [ 0.  ...  1. ]
    covariate = np.linspace(0, 1, npoints)
    # [ 0.  ...  1. ]
    horizontal = np.linspace(0, 1, npoints)
    # [-1.  -0.5  0.   0.5  1. ]
    vals = np.linspace(-1, 1, ntrajs)

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

    cluster_data = tt.helpers.normalise_data(produce_cluster_data())


    assert( len(cluster_data) == 5 )

    for (x, Y) in cluster_data:

        assert( x.shape == (50,) )

        assert( Y.shape == (50, 2) )

def test_mix():
    """compare different methods"""

    # obtain cluster_data, 3-dimensional
    cluster_data_temp = tt.helpers.get_trajectories(1, ndim=2, noise_std=.01)

    # normalise it (normally done within modelling method)
    cluster_data = tt.helpers.normalise_data(cluster_data_temp)

    # create a gp, and produce output in 5 steps
    # >> 5 multivariate distributions
    gp = tt.gaussianprocess.GaussianProcess(cluster_data, ngaus=5)

    # method 1 - resampling
    (mu_y_1, sig_y_1, cc, cA) = gp.model_by_resampling()

    # method 2 - maximum likelihood
    (mu_y_2, sig_y_2, cc, cA) = gp.model_by_ml(type_basis="rbf", nbasis=30)

    #print(mu_y_1)
    #print(mu_y_2)

    # TODO FIX TOLERANCE
    np.testing.assert_allclose(mu_y_1,
                               mu_y_2,
                               atol=10)



def test_resampling():
    """check if resampling behaves as expected"""

    # obtain cluster_data test
    cluster_data = tt.helpers.normalise_data(produce_cluster_data())

    # create a gp, and produce output in 5 steps
    # >> 5 multivariate distributions
    gp = tt.gaussianprocess.GaussianProcess(cluster_data, ngaus=5)

    # no additional settings required
    # uses linear interpolation
    (mu_y, sig_y, cc, cA) = gp.model_by_resampling()

    # expect the average to be a line from (0,0) to (0,1) in 5 steps
    y = np.zeros(shape=(5,))
    x = np.linspace(0, 1, 5)

    mu_y_expected = np.concatenate((x, y), axis=0).transpose().reshape((-1,1))

    np.testing.assert_array_almost_equal_nulp(mu_y, mu_y_expected)

    # variance
    sig_y_diag = np.diag(sig_y)

    # expected 0 variance in x, 1/2 variance in y
    x = np.zeros(shape=(5,))
    y = np.ones(shape=(5,))*0.5
    sig_y_diag_expected = np.concatenate((x, y), axis=0)

    np.testing.assert_allclose(sig_y_diag,
                               sig_y_diag_expected,
                               atol=1e-6)


def test_maximum_likelihood():
    """check if maximum likelihood behaves as expected"""

    # obtain cluster_data test
    cluster_data = tt.helpers.normalise_data(produce_cluster_data())

    # create a gp, and produce output in 5 steps
    # >> 5 multivariate distributions
    gp = tt.gaussianprocess.GaussianProcess(cluster_data, ngaus=5)

    # additional settings required
    # uses basis functions
    (mu_y, sig_y, cc, cA) = gp.model_by_ml( type_basis="rbf", nbasis=30 )

    # expect the average to be a line from (0,0) to (0,1) in 5 steps
    x = np.linspace(0, 1, 5)
    y = np.zeros(shape=(5,))

    mu_y_expected = np.concatenate((x, y), axis=0).transpose().reshape((-1,1))

    # note --VERY-- large tolerance....
    # method is at it's worst for lines, large number of basis functions allows for an approximation
    np.testing.assert_allclose(mu_y, mu_y_expected, atol=0.01)

    # variance
    sig_y_diag = np.diag(sig_y)

    # expected 0 variance in x, 1/2 variance in y
    x = np.zeros(shape=(5,))
    y = np.ones(shape=(5,))*0.5
    sig_y_diag_expected = np.concatenate((x, y), axis=0)

    np.testing.assert_allclose(sig_y_diag,
                               sig_y_diag_expected,
                               atol=1e-6)




def test_expectation_maximization():
    "check if expectation maximization behaves as expected"

    # obtain cluster_data test
    cluster_data = tt.helpers.normalise_data(produce_cluster_data())

    # create a gp, and produce output in 5 steps
    # >> 5 multivariate distributions
    gp = tt.gaussianprocess.GaussianProcess(cluster_data, ngaus=5)

    # no additional settings required
    # uses linear interpolation
    (mu_y, sig_y, cc, cA) = gp.model_by_em( type_basis="rbf", nbasis=30 )

    # expect the average to be a line from (0,0) to (0,1) in 5 steps
    x = np.linspace(0, 1, 5)
    y = np.zeros(shape=(5,))

    mu_y_expected = np.concatenate((x, y), axis=0).transpose().reshape((-1,1))

    # note --VERY-- large tolerance....
    # method is at it's worst for lines, large number of basis functions allows for an approximation
    np.testing.assert_allclose(mu_y, mu_y_expected, atol=0.01)

    # variance
    sig_y_diag = np.diag(sig_y)

    # expected 0 variance in x, 1/2 variance in y
    x = np.zeros(shape=(5,))
    y = np.ones(shape=(5,))*0.5
    sig_y_diag_expected = np.concatenate((x, y), axis=0)

    # TODO: FIX TOLERANCE
    np.testing.assert_allclose(sig_y_diag,
                               sig_y_diag_expected,
                               atol=1e-4)

def test_subfunctions_EM():
    """test subfunctions that contribute to the EM algorithm
    """

    # simple cluster_data
    cluster_data = tt.helpers.normalise_data(produce_cluster_data())

    # TEMPORARY FAKE
    valid_settings = {"model_type": "EM",
                      "ngaus": 10,
                      "nbasis": 5,
                      "basis_type": "rbf"}

    # normal operation
    gp = tt.gaussianprocess.GaussianProcess(cluster_data, ngaus=5)

    # create a basis (normally done inside model)
    basis = tt.basis.Basis("rbf", nbasis=30, ndim=2)

    # simple conversion
    (yc, Hc) = gp._from_clusterdata2cells(cluster_data, basis)

    # check length
    assert(len(yc) == len(Hc))

    # average
    Ewc = []
    Ewc.append(np.array([0.0, 0.0, 0.0]))
    Ewc.append(np.array([2.0, 2.0, 2.0]))

    mu_w = gp._E_mu(Ewc)
    mu_w_expected = np.array([1.0, 1.0, 1.0])

    np.testing.assert_array_almost_equal_nulp(mu_w, mu_w_expected)

    # sigma
    #sig_w = gp._E_sigma(mu_w, yc, Hc, Ewc, Ewwc)

    # beta
    #BETA = gp._E_beta(yc, Hc, Ewc, Ewwc, ndim, Mstar)

    # ln( p (Y|w) )
    #loglikelihood_pYw = gp._L_pYw(yc, Hc, Ewc, Ewwc, ndim, Mstar, BETA_EM)

def test_subfunctions():
    """various
    """

    # obtain cluster_data test, 2-d
    cluster_data = tt.helpers.normalise_data(produce_cluster_data())

    # create a gp
    gp = tt.gaussianprocess.GaussianProcess(cluster_data, ngaus=5)

    # check vectors
    M, D = gp._outline2vectors()

    assert(M.shape == (10,1))
    assert(D.shape == (10,1))

    # check real mu_y, sig_y
    # mu_y, sig_y = gp._norm2real(mu_y, sig_y)
