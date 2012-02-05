#!/usr/bin/env python

from distutils.core import setup, Extension
from numpy.distutils.system_info import get_info

module = Extension('engine',
                   sources = ['engine.cpp'],
                   extra_compile_args = ['-O3'])

setup(name = 'engine',
      version = '1.0',
      description = 'A Pentago playing engine',
      ext_modules = [module])
