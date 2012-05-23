#!/usr/bin/env python

from __future__ import division
from other.core import *
from interface import *
import tempfile

def test_supertensor():
  file = tempfile.NamedTemporaryFile(prefix='test',suffix='.pentago')
  supertensor_test(file.name)

if __name__=='__main__':
  test_supertensor()
