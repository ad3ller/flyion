from setuptools import setup

setup(name='flyion',
      version='0.0.1',
      description='Calculate the trajectory of a charged particle in an electric field',
      author='Adam Deller',
      author_email='a.deller@ucl.ac.uk',
      license='BSD',
      packages=['flyion'],
      install_requires=[
          'scipy>=0.14','numpy>=1.10','pandas>=0.17', 'tqdm>=3.1.4'
      ],
      include_package_data=False,
      zip_safe=False)
