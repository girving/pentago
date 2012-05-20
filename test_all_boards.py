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
    approx_sizes = [1,9,108,777,6114,35515,221421,1106982,5720853,24387986,104415100]
    approx_hashes = [8,1242124013316135169,4596365112089932754,3766756570253749938,-5385124322950766719,-8464112590199284450,-2549407299935802811,441659361128612490,2377047564873664112,157682083666922903,4533891953974912788]
    simple_sizes = [1,3,36,286,2816,15772,105628,565020]
    simple_hashes = [8,410212465185018089,3750615323228832946,4453299882991147388,831431775888651986,-4529085327117717784,4461946080405445225,-2486816462738686179,-7520427304258638725,-3660154264915719466,-9135464216135215359]
    generator = ApproximateBoards()
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
      # Nonsimple
      approx = generator.list(n)
      h = ahash(approx)
      print 'approx: count = %d, hash = %d, ratio %g'%(len(approx),h,len(approx)/sizes[n])
      assert sizes[n]<=len(approx)==approx_sizes[n]
      assert approx_hashes[n]==h
      approx.sort()
      assert generator.is_subset(boards,approx)
      # Simple
      approx = generator.simple_list(n)
      h = ahash(approx)
      print 'simple: count = %d, hash = %d, ratio %g'%(len(approx),h,len(approx)/sizes[n])
      assert sizes[n]<=len(approx)==simple_sizes[n]
      assert simple_hashes[n]==h
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
  approx = ApproximateBoards()
  for n in xrange(0,37):
    steps = min(100000,supercount_boards(n))
    print 'simple_test: n = %d, steps = %d'%(n,steps)
    approx.simple_test(n,steps)

if __name__=='__main__':
  test_approx()
  test_all_boards_super()
