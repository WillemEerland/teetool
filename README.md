[![Travis CI](https://travis-ci.com/WillemEerland/teetool.svg?token=vgGUTGsaoutqpevkkMq4&branch=master)]

# teetool
a package to support with the statistical analysis of trajectory data

# setup the environment via Anaconda
conda create -q -y -n teetool-env python=2.7 --file requirements_conda.txt
source activate teetool-env
pip install --requirement requirements_pip.txt

# build
python setup.py install

# run tests
py.test

# run examples
python /example/example_toy_trajectories_2d.py
python /example/example_toy_trajectories_3d.py

[![Input](https://www.southampton.ac.uk/~wje1n13/teetool/1_input.png)]

[![Output](https://www.southampton.ac.uk/~wje1n13/teetool/2_result.png)]


