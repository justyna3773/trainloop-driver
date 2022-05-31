import re
from setuptools import setup, find_packages

setup(name='trainloop-driver',
      packages=[package for package in find_packages()
                if package.startswith('driver')],

      install_requires=[
          'gym',
          'scipy',
          'tqdm',
          'joblib',
          'cloudpickle',
          'click',
          'opencv-python',
          'matplotlib',
          'pandas',
          'gym_cloudsimplus',
          'stable_baselines3',
      ],
      description='Training loop',
      author='Pawel Koperek',
      url='https://gitlab.com/pkoperek/trainingloop-driver',
      author_email='pkoperek@gmail.com',
      version='0.1.1',
      entry_points={
          'console_scripts': ['trainloop=run:main'],
      }
      )
