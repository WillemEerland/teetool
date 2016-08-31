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

[![2d toy 0](https://www.southampton.ac.uk/~wje1n13/teetool/2d_toy0.png)]

[![2d toy 1](https://www.southampton.ac.uk/~wje1n13/teetool/2d_toy1.png)]

[![2d inter](https://www.southampton.ac.uk/~wje1n13/teetool/2d_toy0_toy1.png)]

python example/example_toy_trajectories_3d.py

[![3d toy 0](https://www.southampton.ac.uk/~wje1n13/teetool/3d_toy0.png)]

[![3d toy 1](https://www.southampton.ac.uk/~wje1n13/teetool/3d_toy1.png)]

[![3d inter](https://www.southampton.ac.uk/~wje1n13/teetool/3d_toy0_toy1.png)]
