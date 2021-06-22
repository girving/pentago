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
          'aiofiles>=0.7.0',
          'chardet<4.0',  # For gcloud-aio
          'dm-haiku>=0.0.3',
          'gcloud-aio-storage>=6.1.0',
          'jax>=0.2.12',
          'optax>=0.0.6',
          'pytest-asyncio>=0.15.1',
          'requests>=2.25.1',
      ],
)
