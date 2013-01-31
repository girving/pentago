#!/usr/bin/env python

from __future__ import division
from other.core import *
from pentago import *
import hashlib

def helper(symmetries):
  print
  def portable_hash(boards):
    return hashlib.sha1(boards.astype(boards.dtype.newbyteorder('<')).tostring()).hexdigest()
  if symmetries==1:
    sizes = [1,36,1260,21420,353430,3769920,38955840]
    hashes = '05fe405753166f125559e7c9ac558654f107c7e9 6f436e8746348d7a0a2f9416323e0ac9f7ee8992 5ffad230681a476cbc33901b8ed7d473e721b995 5e6eda2c60533b0eed85113e33ceb07289a7aef2 15e8125d1798b6e18f5d02691c5a5123ce72f3b2 499d9155a02f8a07e5c7314dd5dcc1561cd059d6 f392633216cbeaeb22e7073f2d953805509b168f'.split()
  elif symmetries==8:
    sizes = [1,6,165,2715,44481,471870,4871510,36527160,264802788]
    hashes = '05fe405753166f125559e7c9ac558654f107c7e9 2f3618ae0973fd076255bf892cb1594470b203ef a59b31ec91dc175aab1bfd94fb8e4ff5a8fc1fe0 1d72107a01b55d13b13e509cf700f31ff2d62990 5322f3629de31817a217666ac4bc197577b06ca3 208ca65e8cb26c691f6ec14bf398d1dbec92ccb0 7b79248adf142680e1ae37a54ebbeb3e30491458 fe422248b4f41a49f8dbc884f49c39eb253fb2c6 af40db6c2575017dc62f9837f1055a9cb12e87c2'.split()
  elif symmetries==2048:
    sizes = [1,3,30,227,2013,13065,90641,493844,2746022,12420352,56322888]
    hashes = '05fe405753166f125559e7c9ac558654f107c7e9 597241c65bdbdde28269031b3035bfff95c8dfa4 bedc958370b06c3951a3a55010801ab1b53a36af 09dfa654eaaccfca1c50d08ee21d23ccdbe737f2 c1982d1252ae8209b9bd4fa7ffed05708a72e648 764e7486ff3c1955eb5967c896e7cdb3778c0dd3 60ddd31f7c7b3e5817c07c952eebb8e350efc6c2 75e740e79eab6f43a3f22c081c37916333246ae7 cba975b696a4b2a0f6dbb0ad23507009a47af2c9 345f651e20f0ac9e2a11872f0d574858d53f7963 60cb22beee25e8a3d1ac548de6247bbba70d7298'.split()
    all_sizes = [1,3,36,286,2816,15772,105628,565020,3251984]
    all_hashes = '05fe405753166f125559e7c9ac558654f107c7e9 597241c65bdbdde28269031b3035bfff95c8dfa4 a6def1687498ba310f0f33dda4c3de1b845ba8ae 2464973e1ef7bde09ba5eef9fe65731955371590 1c301e5953ae69d37781d46b9fe104f9c8371491 212f49c5171495c2ff93fd0ca71d2c27e31ede45 709cf5ada8411202a55616b68ca40649c9bdbfab 8275ac68db3dde43e25c90da19e57d3ec932f907 9b32620990c2a784b9fdbb0c6c89caa51b68b5b4'.split()
    assert len(all_sizes)==len(all_hashes)
  assert len(sizes)==len(hashes)
  for n in xrange(5 if symmetries==1 else 6):
    boards = all_boards(n,symmetries)
    if 0 and n<3:
      print '\n\n-------------------------------- symmetries %d, n %d, count %d ---------------------------------\n'%(symmetries,n,len(boards))
      for b in boards:
        print show_board(b)
    h = portable_hash(boards)
    print 'n = %d, count = %d, hash = %s'%(n,len(boards),h)
    assert sizes[n]==len(boards)
    assert hashes[n]==h
    if symmetries==2048 and n<len(all_sizes):
      boards.sort()
      assert len(boards)==count_boards(n,2048)
      assert len(unique(boards))==len(boards)
      approx = all_boards_list(n)
      h = portable_hash(approx)
      print 'approx: count = %d, hash = %s, ratio %g'%(len(approx),h,len(approx)/sizes[n])
      assert sizes[n]<=len(approx)==all_sizes[n]
      assert all_hashes[n]==h
      approx = superstandardize(approx)
      approx.sort()
      assert sorted_array_is_subset(boards,approx)
      print

def test_all_boards_raw():
  helper(1)

def test_all_boards():
  helper(8)

def test_all_boards_super():
  helper(2048)

def test_small_hashes():
  # Verify that boards with at most 2 stones all map to different hashes mod 511.
  # This ensures that such positions will never disappear from the transposition table as the result of a collision.
  boards = hstack([all_boards(n,2048) for n in 0,1,2])
  assert distinguishing_hash_bits(boards)==9

def test_sample():
  for n in xrange(0,37):
    steps = min(100000,count_boards(n,2048))
    print 'sample test: n = %d, steps = %d'%(n,steps)
    all_boards_sample_test(n,steps)

def test_rmin():
  rmin_test()

if __name__=='__main__':
  rmin_test()
  test_sample()
  test_all_boards_super()
