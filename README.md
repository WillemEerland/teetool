[![Travis CI](https://travis-ci.org/WillemEerland/teetool.svg?branch=master)](https://travis-ci.org/WillemEerland/teetool)

# teetool
a package to support with the statistical analysis of trajectory data -- helpful at determining the probability of clusters (collection) of trajectories colliding

purely spatial, ignores temporal effects

documentation is available at https://willemeerland.github.io/teetool/

# setup the environment in Linux

- download & install Anaconda from https://www.continuum.io/download
- open terminal
- navigate to Teetool directory

> conda create -n teetool python=2.7 pytest pytest-cov mayavi numpy scipy matplotlib pyside

> source activate teetool

> pip install .

# setup the environment in macOS

as matplotlib requires a backend, it is easiest to replace

> pip install matplotlib

with

> conda install matplotlib

# setup the environment in Windows

- download & install Anaconda from https://www.continuum.io/download
- open 'Anaconda prompt'
- navigate to Teetool directory

> conda create -n teetool python=2.7 pytest pytest-cov mayavi numpy scipy matplotlib pyside

> activate teetool

> set QT_API=pyside

> pip install .

# run tests

> py.test

# run tests (including coverage report)

> (cd test ; py.test -v --cov-report html --cov=teetool)

# run examples via Jupyter notebook

> pip install .[example]

> jupyter notebook

- find example/ in browser and run files in order
