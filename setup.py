#

from setuptools import setup

setup(name='teetool',
      version='1.0',
      description='trajectory analysis tool',
      author='Willem Eerland',
      author_email='w.j.eerland@soton.ac.uk',
      packages=['teetool'],
      install_requires=['numpy',
                        'scipy',
                        'matplotlib',
                        'mayavi'],
      extras_require={'example': ['notebook']},
      python_requires='>=3.6'
      )
