#!/usr/bin/env python

from __future__ import division
from geode import *
from pentago import *
import tempfile

def test_thread_pool():
  init_threads(-1,-1)
  thread_pool_test()

if __name__=='__main__':
  test_thread_pool()
