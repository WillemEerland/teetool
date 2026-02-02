[![Travis CI](https://travis-ci.org/WillemEerland/teetool.svg?branch=master)](https://travis-ci.org/WillemEerland/teetool)

# teetool
a package to support with the statistical analysis of trajectory data -- helpful at determining the probability of clusters (collection) of trajectories colliding

purely spatial, ignores temporal effects

publication is available at http://doi.org/10.5334/jors.163

documentation is available at https://willemeerland.github.io/teetool/

# setup the environment in Linux << NOT CHECKED FOR LATEST VERSION >>

- download & install Anaconda from https://www.continuum.io/download
- open terminal
- navigate to Teetool directory

> conda create -n teetool python=3 pytest pytest-cov mayavi numpy scipy matplotlib

> source activate teetool

> pip install .

# setup the environment in macOS << NOT CHECKED FOR LATEST VERSION >>

- download & install Anaconda from https://www.continuum.io/download
- open terminal
- navigate to Teetool directory

> conda create -n teetool python=3 pytest pytest-cov mayavi numpy scipy matplotlib

> source activate teetool

> pip install .

# setup the environment in Windows << CHECKED >>

- download & install Anaconda from https://www.continuum.io/download
- open 'Anaconda prompt'
- navigate to Teetool directory

> conda create -n teetool python=3 pytest pytest-cov mayavi numpy scipy matplotlib

> activate teetool

> pip install .

# run tests

> py.test

# run tests, including coverage report

> (cd test ; py.test -v --cov-report html --cov=teetool)

# run examples via Jupyter notebook

> pip install .[example]

> jupyter notebook

- find example/ in browser and run files in order
