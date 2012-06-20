#!/usr/bin/env python

from __future__ import division
from other.core import *
from interface import *

def random_supers(n,):
  # Note: biased towards black wins (shouldn't matter for testing)
  data = fromstring(random.bytes(64*n),uint64).reshape(n,2,4)
  data[:,1] &= ~data[:,0]
  return data

def test_interleave():
  random.seed(8428121)
  for _ in xrange(10):
    src = random_supers(random.randint(1000,1200))
    dst = src.copy() 
    interleave(dst)
    uninterleave(dst)
    assert all(src==dst) 

def test_compact():
  random.seed(8428123)
  for _ in xrange(10):
    src = random_supers(random.randint(1000,1200))
    dst = compact(src.copy())
    assert len(dst)==ceil(32*src.size/5)
    dst = uncompact(dst)
    assert all(src==dst)

if __name__=='__main__':
  test_compact()
  test_interleave()
