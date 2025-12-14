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
          'aiofiles<24.0.0,>=0.7.0',  # For gcloud-aio-storage
          'chardet<4.0',  # For gcloud-aio
          'dm-haiku>=0.0.3',
          'flax>=0.12.1',
          'gcloud-aio-storage>=9.3.0',
          'jax>=0.8.1',
          'optax>=0.0.6',
          'pytest>=8.3.5',
          'pytest-asyncio>=0.15.1',
          'requests>=2.25.1',
          'wandb',
      ],
      packages = [],
)
