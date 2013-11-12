#!/usr/bin/env python

from setuptools import setup,find_packages

setup(
  # Basics
  name='pentago',
  version='0.0-dev',
  description='A brute force pentago solver',
  author='Geoffrey Irving',
  author_email='irving@naml.us',
  url='http://github.com/girving/pentago',

  # Installation
  packages=find_packages(),
  package_data={'geode':['*.py','*.so']},
)
