"""
<description>
"""

import numpy as np
import pytest as pt

import teetool as tt


def test_init():
    """
    <description>
    """

    # test 1
    # set values
    name_1 = "Hello world!"
    D_1 = 2
    world_1 = tt.World(name=name_1, ndim=D_1)
    assert (world_1._name == name_1)
    assert (world_1._ndim == D_1)

    # test 2
    # default values
    world_2 = tt.World()
    assert (world_2._name == "")
    assert (world_2._ndim == 3)

    # test 3
    # bad name
    name_3 = 5
    with pt.raises(TypeError) as testException:
        world_3 = tt.World(name_3)

    # bad TYPE dimension
    name_4 = "Hello world!"
    D_4 = "Hello World!"
    with pt.raises(TypeError) as testException:
        world_4 = tt.World(name_4, D_4)

    # bad VALUE dimension
    name_5 = "Hello World!"
    D_5 = 1
    with pt.raises(ValueError) as testException:
        world_5 = tt.World(name_5, D_5)


def test_addCluster():
    """
    <description>
    """

    mdim = 3  # dimensionality

    # build world
    world_1 = tt.World(name="test", ndim=mdim)

    # build a valid cluster
    correct_cluster_name = "correct data"

    # normal operation
    for mtype in [0, 1]:
        correct_cluster_data = tt.helpers.get_trajectories(mtype, ndim=mdim, ntraj=5)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    #
    these_clusters = world_1.getCluster()

    assert (len(these_clusters) == 2)

    #
    wrong_cluster_name = 5
    with pt.raises(TypeError) as testException:
        world_1.addCluster(correct_cluster_data, wrong_cluster_name)

    #
    wrong_cluster_data = 5
    with pt.raises(TypeError) as testException:
        world_1.addCluster(wrong_cluster_data, correct_cluster_name)

    #
    wrong_trajectory_data = 5
    wrong_cluster_data = correct_cluster_data
    wrong_cluster_data.append(wrong_trajectory_data)
    with pt.raises(TypeError) as testException:
        world_1.addCluster(wrong_cluster_data, correct_cluster_name)


def test_model():
    """
    tests the modelling functionality
    """

    # build world
    world_1 = tt.World(name="model test", ndim=3)

    # add trajectories
    for ntype in [0, 1]:
        correct_cluster_name = "toy {0}".format(ntype)
        correct_cluster_data = tt.helpers.get_trajectories(ntype, ndim=3, ntraj=20)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    # model all trajectories
    settings = {}
    settings["model_type"] = "resampling"
    settings["ngaus"] = 10

    # build a model
    world_1.buildModel(settings)

    with pt.raises(TypeError) as testException:
        world_1.buildModel(settings, "Hello World!")

    with pt.raises(ValueError) as testException:
        world_1.buildModel(settings, [-1])

    # log-likelihood
    (ss_list, [xx, yy, zz]) = world_1.getLogLikelihood([0, 1])

    with pt.raises(TypeError) as testException:
        (ss_list, [xx, yy, zz]) = world_1.getLogLikelihood("Hello World!")

    with pt.raises(ValueError) as testException:
        (ss_list, [xx, yy, zz]) = world_1.getLogLikelihood([-1])

    # build tube (twice!)
    for i in range(2):
        (ss_list, [xx, yy, zz]) = world_1.getTube([0, 1])


    # complexity map
    ss, xx, yy, zz = world_1.getComplexityMap([0, 1])

    # overview
    world_1.overview()

    # produce some numbers
    world_1.getTubeStats()

    # clear world
    world_1.clear()

def test_data_point():
    """obtain a point cloud based on the data
    """

    # build world
    world = tt.World(name="model test", ndim=3)

    # test (x, Y)
    x = np.linspace(0,1,100)
    Y = np.zeros((100, 3)) # << dimension
    Y[:,0] = x
    Y[:,1] = x
    Y[:,2] = x

    a = world._get_point_from_xY(x, Y, x1=0.0)
    a_expected = np.array([0.0, 0.0, 0.0])
    np.testing.assert_array_almost_equal_nulp(a, a_expected)

    a = world._get_point_from_xY(x, Y, x1=1.0)
    a_expected = np.array([1.0, 1.0, 1.0])
    np.testing.assert_array_almost_equal_nulp(a, a_expected)


    # test cluster_data
    cluster_data = []
    cluster_data.append ( (x, Y) ) # 1
    cluster_data.append ( (x, Y) ) # 2
    cluster_data.append ( (x, Y) ) # 3
    cluster_data.append ( (x, Y) ) # 4

    A = world._get_point_from_cluster_data(cluster_data, x1=0.0)
    A_expected = np.zeros((4,3))
    np.testing.assert_array_almost_equal_nulp(A, A_expected)

    A = world._get_point_from_cluster_data(cluster_data, x1=1.0)
    A_expected = np.ones((4,3))
    np.testing.assert_array_almost_equal_nulp(A, A_expected)

    # add trajectories
    world.addCluster(cluster_data, "1")
    world.addCluster(cluster_data, "2")

    # obtain list of points
    clusterP = world.getClusterPoints( x1 = 0.0 )

    # check values inside clusters
    for i in [0, 1]:
        np.testing.assert_array_almost_equal_nulp(clusterP[i], np.zeros((4,3)))

    # obtain list of points
    clusterP = world.getClusterPoints( x1 = 1.0 )

    # check values
    for i in [0, 1]:
        np.testing.assert_array_almost_equal_nulp(clusterP[i], np.ones((4,3)))
