[![Travis CI](https://travis-ci.com/WillemEerland/teetool.svg?token=vgGUTGsaoutqpevkkMq4&branch=master)]

# teetool
a package to support with the statistical analysis of trajectory data -- helpful at determining the probability of clusters (collection) of trajectories colliding

purely spatial, ignores temporal effects

# setup the environment in Linux

conda create -n teetool python=2.7 pytest pytest-cov mayavi numpy scipy

source activate teetool-env

pip install pathos matplotlib

python setup.py install

# run tests (including coverage report)

(cd test ; py.test -v --cov-report html --cov=teetool)

# example/example_toy_2d.py

![2d intersection](https://www.southampton.ac.uk/~wje1n13/teetool/2d_intersection.png)

shows log-probability of intersecting clusters

# example/example_toy_3d.py

![3d toy 0](https://www.southampton.ac.uk/~wje1n13/teetool/3d_input.png)

shows input trajectory data

![3d toy 1](https://www.southampton.ac.uk/~wje1n13/teetool/3d_toy.png)

shows log-probability of a single cluster

![3d inter](https://www.southampton.ac.uk/~wje1n13/teetool/3d_intersection.png)

shows log-probability of two clusters intersecting
