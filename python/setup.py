from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name = 'beat tracker',
      ext_modules = cythonize("*.pyx"),
      include_dirs=[numpy.get_include()])