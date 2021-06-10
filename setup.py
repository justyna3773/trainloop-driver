import re
from setuptools import setup, find_packages

setup(name='trainloop-driver',
      packages=[package for package in find_packages()
                if package.startswith('driver')],

      install_requires=[
          'gym>=0.15.4, <0.18.0',
          'scipy',
          'tqdm',
          'joblib',
          'cloudpickle==1.2.0',
          'click',
          'opencv-python',
          'matplotlib',
          'pandas',
          'gym_cloudsimplus',
          'stable_baselines3[extra]',
      ],
      description='Training loop for the PPO2 policy',
      author='Pawel Koperek',
      url='https://gitlab.com/pkoperek/trainingloop-driver',
      author_email='pkoperek@gmail.com',
      version='0.1.0',
      entry_points={
          'console_scripts': ['trainloop=driver.run:main'],
      }
      )
