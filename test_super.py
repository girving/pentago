#!/usr/bin/env python

from __future__ import division
from interface import *

def test_wins():
  try:
    engine.super_win_test(100000)
  except AssertionError,e:
    m = re.match('side (\d+), rside (\d+)',str(e))
    side = int(m.group(1))
    rside = int(m.group(2))
    q = asarray(side,uint64).reshape((1,)).view(int16)
    print 'side %d, quadrants %d %d %d %d\n%s'%(side,q[0],q[1],q[2],q[3],show_side(side))
    print 'rside %d\n%s'%(rside,show_side(rside))
    raise

def test_rmax():
  engine.super_rmax_test(100000)

def test_bool():
  engine.super_bool_test()

if __name__=='__main__':
  test_wins()
  test_rmax()
  test_bool()
