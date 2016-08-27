# teetool
a package to support with the statistical analysis of trajectory data

# setup the environment via Anaconda
conda create -n teetool-env python=2.7 pytest mayavi
source activate teetool-env

# run tests
py.test test

# run example
python /example/*.py [choose a file here]
