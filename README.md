[![Travis CI](https://travis-ci.com/WillemEerland/teetool.svg?token=vgGUTGsaoutqpevkkMq4&branch=master)]

# teetool
a package to support with the statistical analysis of trajectory data

# setup the environment via Anaconda

conda create -n teetool-env python=2.7 pytest mayavi

source activate teetool-env

# run tests
py.test test

# run example

python setup.py install

python /example/example_toy_trajectories_3d.py

[![Input](https://www.southampton.ac.uk/~wje1n13/teetool/1_input.png)]

[![Output](https://www.southampton.ac.uk/~wje1n13/teetool/2_result.png)]


