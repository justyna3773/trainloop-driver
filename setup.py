import re
from setuptools import setup, find_packages


setup(name='trainloop-driver',
      packages=[package for package in find_packages()
                if package.startswith('driver')],
      install_requires=[
          'gym>=0.15.4, <0.16.0',
          'scipy',
          'tqdm',
          'joblib',
          'cloudpickle==1.2.0',
          'click',
          'opencv-python',
          'matplotlib',
          'pandas',
          'gym_cloudsimplus',
      ],
      description='Training loop for the PPO2 policy',
      author='Pawel Koperek',
      url='https://gitlab.com/pkoperek/trainingloop-driver',
      author_email='pkoperek@gmail.com',
      version='0.1.0',
      # entry_points = {
      #     'console_scripts': ['trainloop=driver.run:main'],
      # }
      entry_points = {
            'console_scripts': ['trainloop=run:main'],
      }

)


# ensure there is some tensorflow build with version above 1.4
import pkg_resources
tf_pkg = None
for tf_pkg_name in ['tensorflow', 'tensorflow-gpu', 'tf-nightly', 'tf-nightly-gpu']:
    try:
        tf_pkg = pkg_resources.get_distribution(tf_pkg_name)
    except pkg_resources.DistributionNotFound:
        pass
assert tf_pkg is not None, 'TensorFlow needed, of version above 1.4'
from distutils.version import LooseVersion
assert LooseVersion(re.sub(r'-?rc\d+$', '', tf_pkg.version)) >= LooseVersion('1.4.0')
