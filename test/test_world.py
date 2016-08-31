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
    world_1 = tt.World(name=name_1, dimension=D_1)
    assert (world_1._name == name_1)
    assert (world_1._D == D_1)

    assert (world_1.overview() == True)

    # test 2
    # default values
    world_2 = tt.World()
    assert (world_2._name == "")
    assert (world_2._D == 3)

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

    D = 3  # dimensionality

    # build world
    world_1 = tt.World(name="test", dimension=D)

    # build a valid cluster
    correct_cluster_name = "correct data"

    # normal operation
    for ntype in [1, 2]:
        correct_cluster_data = tt.helpers.get_trajectories(ntype, D, N=5)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    #
    these_clusters = world_1.getClusters()

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
    world_1 = tt.World(name="model test", dimension=3)

    # add trajectories
    for ntype in [0, 1]:
        correct_cluster_name = "toy {0}".format(ntype)
        correct_cluster_data = tt.helpers.get_trajectories(ntype, D=3, N=20)
        world_1.addCluster(correct_cluster_data, correct_cluster_name)

    # model all trajectories
    settings = {}
    settings["model_type"] = "resample"
    settings["mgaus"] = 10

    # build a model
    world_1.buildModel(0, settings)

    with pt.raises(TypeError) as testException:
        world_1.buildModel("Hello World!", settings)

    with pt.raises(ValueError) as testException:
        world_1.buildModel(-1, settings)

    # build log-likelihood
    world_1.buildLogProbality(0)

    with pt.raises(TypeError) as testException:
        world_1.buildLogProbality("Hello World!")

    with pt.raises(ValueError) as testException:
        world_1.buildLogProbality(-1)

    assert (world_1.overview() == True)

    world_1.setResolution(10,10,10)
