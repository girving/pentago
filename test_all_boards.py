#!/usr/bin/env python

from __future__ import division
from other.core import *
from interface import *

def all_boards_test(symmetries):
  print
  if symmetries==1:
    sizes = [1,36,1260,21420,353430,3769920,38955840]
    hashes = [8,8510287199477074976,8942154448276293432,8070288876157080644,-3977280848011655236,719338433904157812,-3162374635518058044]
  elif symmetries==8:
    sizes = [1,6,165,2715,44481,471870,4871510,36527160,264802788]
    hashes = [8,-6878000688188254263,-2173867943958208377,-4633991874873228126,-683072618870330000,4414072784886068265,-713043673212511325,535474690767935483,5651753791106627528]
  elif symmetries==2048:
    sizes = [1,3,30,227,2013,13065,90641,493844,2746022,12420352,56322888]
    hashes = [8,410212465185018089,-5518763455213967035,4939794567825530384,4040790906611107384,1937332596658284469,-6778473604366810076,4708544786165225203,6204346775015887223,-1601224996803759779,-5782498499478641186]
    approx_sizes = [1,3,36,286,2816,15772,105628,565020]
    approx_hashes = [8,410212465185018089,-1662608180067154078,-7047827481926299996,2843504063880042090,6662376299248310788,-5638710477457538887,6279152328712157581,34579930516338039,259199675726595994,-4927414379464215279]
    generator = AllBoards()
  assert len(sizes)==len(hashes)
  for n in xrange(5 if symmetries==1 else 6):
    boards = all_boards(n,symmetries)
    if 0 and n<3:
      print '\n\n-------------------------------- symmetries %d, n %d, count %d ---------------------------------\n'%(symmetries,n,len(boards))
      for b in boards:
        print show_board(b)
    h = ahash(boards)
    print 'n = %d, count = %d, hash = %d'%(n,len(boards),h)
    assert sizes[n]==len(boards)
    assert hashes[n]==h
    if symmetries==2048:
      boards.sort()
      assert len(boards)==supercount_boards(n)
      assert len(unique(boards))==len(boards)
      approx = generator.list(n)
      h = ahash(approx)
      print 'approx: count = %d, hash = %d, ratio %g'%(len(approx),h,len(approx)/sizes[n])
      assert sizes[n]<=len(approx)==approx_sizes[n]
      assert approx_hashes[n]==h
      approx = superstandardize(approx)
      approx.sort()
      assert generator.is_subset(boards,approx)
      print

def test_all_boards_raw():
  all_boards_test(1)

def test_all_boards():
  all_boards_test(8)

def test_all_boards_super():
  all_boards_test(2048)

def test_small_hashes():
  # Verify that boards with at most 2 stones all map to different hashes mod 511.
  # This ensures that such positions will never disappear from the transposition table as the result of a collision.
  boards = hstack([all_boards(n,2048) for n in 0,1,2])
  assert distinguishing_hash_bits(boards)==9

def test_approx():
  approx = AllBoards()
  for n in xrange(0,37):
    steps = min(100000,supercount_boards(n))
    print 'approx test: n = %d, steps = %d'%(n,steps)
    approx.test(n,steps)

if __name__=='__main__':
  test_approx()
  test_all_boards_super()
