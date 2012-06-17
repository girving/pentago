#!/usr/bin/env python

from __future__ import division
from other.core import *
from interface import *
import tempfile

def test_thread_pool():
  init_thread_pools(-1,-1)
  thread_pool_test()

if __name__=='__main__':
  test_thread_pool()
