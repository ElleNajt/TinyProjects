# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 15:47:35 2020

@author: lnajt
"""


from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("primes.pyx", annotate=True)
)