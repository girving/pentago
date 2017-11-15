#!/usr/bin/env python

from __future__ import division
from geode import *
from pentago import *

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

def test_wavelet():
  random.seed(9847224)
  for s0 in 3,8:
    for s1 in 3,8:
      for s2 in 3,8:
        for s3 in 3,8:
          shape = s0,s1,s2,s3
          data = random_supers(product(shape)).reshape(shape+(2,4))
          save = data.copy()
          wavelet_transform(data)
          wavelet_untransform(data) 
          assert all(data==save)

if __name__=='__main__':
  test_wavelet()
  test_compact()
  test_interleave()
