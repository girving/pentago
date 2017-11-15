#!/usr/bin/env python

from __future__ import division,print_function,unicode_literals
from pentago import *
from geode import *

def random_board(stones):
  assert 0<=stones<=36
  flat = zeros(36,dtype=int32)
  entries = arange(36)
  random.shuffle(entries)
  flat[entries[:stones//2]] = 2
  flat[entries[stones//2:stones]] = 1
  return from_table(flat.reshape(6,6))

def test_mid_internal():
  random.seed(5554)
  init_supertable(20)
  for slice in range(30,36+1)[::-1]:
    for _ in xrange(16):
      board = random_board(slice)
      for parity in 0,1:
        midsolve_internal_test(board,parity)

def test_mid():
  random.seed(2223)
  init_supertable(20)
  workspace = midsolve_workspace(30)
  empty = empty_block_cache()
  for slice in range(30,35+1):
    for _ in xrange(16):
      root = random_board(slice)
      for middle in 0,1:
        high = high_board_t(root,middle)
        moves = frozenset(high.moves() if middle else [b for a in high.moves() for b in a.moves()])
        values = midsolve(root,middle,[m.board for m in moves],workspace)
        assert len(values)==len(moves)
        for m in moves:
          assert values[m.board]==m.value(empty)

def test_half():
  halfsuper_test(1024)

if __name__=='__main__':
  test_half()
  test_mid_internal()
  test_mid()
