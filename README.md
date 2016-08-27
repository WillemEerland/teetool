# teetool
a package to support with the statistical analysis of trajectory data

# setup the environment
conda create -n teetool-env --file requirements.txt
source activate teetool-env

# run tests
py.test test/

# run example
python /example/*.py [choose a file here]
