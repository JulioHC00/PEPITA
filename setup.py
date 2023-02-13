from setuptools import setup
  
# reading long description from file
# with open('DESCRIPTION.txt') as file:
#     long_description = file.read()
  
  
# specify requirements of your package here
REQUIREMENTS = ['numpy', 'theano', 'exoplanet', 'matplotlib']
  
# some more details
#CLASSIFIERS = [
#    'Topic :: Internet',
#    'License :: OSI Approved :: MIT License',
#    'Programming Language :: Python',
#    'Programming Language :: Python :: 2',
#    'Programming Language :: Python :: 2.6',
#    'Programming Language :: Python :: 2.7',
#    'Programming Language :: Python :: 3',
#    'Programming Language :: Python :: 3.3',
#    'Programming Language :: Python :: 3.4',
#    'Programming Language :: Python :: 3.5',
#    ]
  
# calling the setup function 
setup(name='pepita',
      version='1.0.0',
      description='Predict the precision of exoplanet parameters from transit light-curves using information analysis techniques.',
#      long_description=long_description,
      url='https://github.com/JulioHC00/PEPITA',
      author='Julio Hernandez Camero',
      author_email='julio.camero.21@ucl.ac.uk',
      license='MIT',
      packages=['pepita'],
#      classifiers=CLASSIFIERS,
      install_requires=REQUIREMENTS,
      keywords='exoplanets analysis transits'
      )
