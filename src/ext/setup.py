from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    name='goertzel_cython',
    ext_modules=cythonize("goertzel_cython.pyx"),
    include_dirs=[numpy.get_include()]
)