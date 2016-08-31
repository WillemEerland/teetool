[![Travis CI](https://travis-ci.com/WillemEerland/teetool.svg?token=vgGUTGsaoutqpevkkMq4&branch=master)]

# teetool
a package to support with the statistical analysis of trajectory data

# setup the environment in Linux

conda create -n teetool-env python=2.7 pytest pytest-cov mayavi

source activate teetool-env

pip install pathos matplotlib

python setup.py install

# setup the environment in OSX [2D]

conda create -n teetool-env python=2.7 pytest pytest-cov matplotlib

source activate teetool-env

pip install pathos

python setup.py install

# setup the environment in OSX [3D]

conda create -n teetool-env python=2.7 pytest pytest-cov mayavi

source activate teetool-env

pip install pathos

python setup.py install

# run tests

py.test -v --cov-report html --cov=teetool test/

# run examples

python example/example_toy_trajectories_2d.py

python example/example_toy_trajectories_3d.py

[![Input](https://www.southampton.ac.uk/~wje1n13/teetool/1_input.png)]

[![Output](https://www.southampton.ac.uk/~wje1n13/teetool/2_result.png)]


