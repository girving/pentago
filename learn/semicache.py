#!/usr/bin/env python3
"""Cache that forgets itself when pickled"""

import functools


class semicache:
  def __init__(self, f):
    self.__setstate__(f)

  def __call__(self, *args, **kwds):
    return self._cached(*args, **kwds)

  def __getstate__(self):
    """Ignore the cache when pickling"""
    return self._f

  def __setstate__(self, f):
    self._f = f
    self._cached = functools.cache(f)
