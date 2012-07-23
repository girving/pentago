#!/usr/bin/env python

from __future__ import division
from pentago import *
from other.core.utility import Log

def test_partition():
  partition_test()

if __name__=='__main__':
  Log.configure('test',False,False,100)
  partition_test()
