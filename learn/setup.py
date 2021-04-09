#!/usr/bin/env python3

from setuptools import setup

setup(name='pentago',
      version='0.1',
      description='Pentago neural network training',
      author='Geoffrey Irving',
      author_email='irving@naml.us',
      url='https://perfect-pentago.net',
      license='BSD3',
      install_requires=[
          'dm-haiku',
          'jax',
          'optax',
          'requests',
      ],
)
