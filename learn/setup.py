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
          'apache-beam>=2.30',
          'Cython==0.29.23',
          'dm-haiku>=0.0.3',
          'google-cloud-storage>=1.37.1',
          'jax>=0.2.12',
          'optax>=0.0.6',
          'requests>=2.25.1',
      ],
)
