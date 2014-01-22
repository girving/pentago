#!/usr/bin/env python

from __future__ import division
from geode import *
from pentago import *

randint = random.randint

def randboard():
  return sum(randint(3**9,size=4)<<16*arange(4))

def old_moves(board,turn,simple):
  return [flip_board(m,1-turn) for m in (simple_moves if simple else moves)(flip_board(board,turn))]

def test_misc():
  random.seed(73120)
  for _ in xrange(100):
    board = randboard()
    assert pack(unpack(board))==board
    assert from_table(to_table(board))==board
    assert flip_board(flip_board(board))==board

def win_test(rotations):
  assert rotations in 'none single any'.split()
  verbose = 1
  random.seed(81383)
  def distance(closeness):
    return 6-(closeness>>16)
  def choose(options):
    assert len(options)
    return options[randint(len(options))]
  if verbose:
    def log(name):
      print '%s: closeness %d, distance %d, count %d'%(name,close,dist,close&0xffff)
  else:
    def log(name):
      pass
  if rotations=='none':
    status = pentago_core.status
    closeness = unrotated_win_closeness
    special = ()
  elif rotations=='single':
    def status(board):
      st = rotated_status(board)
      assert st==(len([() for qx in 0,1 for qy in 0,1 for count in -1,1 if pentago_core.status(rotate(board,qx,qy,count))&1])!=0)
      return st
    closeness = rotated_win_closeness
    # In some cases, black can win by rotating the same quadrant in either direction.  Closeness computation
    # counts this as a single "way", even though it is really two, which may make it impossible to white to reduce closeness.
    special = set([3936710634889235564,8924889845,53214872776869257,616430895238742070,45599372358462108,1194863002822977710,
                   3559298137047367680,2485989107724386484,1176595961439925315])
  elif rotations=='any':
    status = arbitrarily_rotated_status
    closeness = arbitrarily_rotated_win_closeness
    special = ()
  wins = ties = 0
  for i in xrange(100): # (256):
    # Start with an empty board
    board = 0
    close = closeness(board)
    dist = distance(close)
    assert dist==5
    log('\nstart')
    if rotations=='none':
      # Give black several free moves to even the odds
      for _ in xrange(randint(15)):
        board = choose(old_moves(board,turn=0,simple=1))
      close = closeness(board)
      dist = distance(close)
      log('seeds')
    while dist:
      # Verify that a well-placed white stone reduces closeness
      try:
        if rotations!='any': # We should almost always be able to find a move reducing closeness
          board,close = choose([(b,c) for b in old_moves(board,turn=1,simple=1) for c in [closeness(b)] if c<close])
        else: # Black has too much freedom to rotate, so don't require a reduction in closeness
          close,_,board = sorted((closeness(b),randint(1000),b) for b in old_moves(board,turn=1,simple=1))[0]
        dist = distance(close)
      except AssertionError:
        if int(board) not in special:
          close = closeness(board)
          dist = distance(close)
          print 'i %d, board %d, distance %d, ways %d\n%s'%(i,board,dist,close&((1<<16)-1),show_board(board))
          raise
      log('after white')
      if dist==6:
        ties += 1
        break
      # Verify that a well-placed black stone reduces distance
      board,close,dist = choose([(b,c,d) for b in old_moves(board,turn=0,simple=1) for c in [closeness(b)] for d in [distance(c)] if d<dist])
      log('after black')
      # Verify that we've won iff distance==0
      assert (dist==0)==(status(board)&1)
      if dist==0:
        wins += 1
        break
  print 'wins %d, ties %d'%(wins,ties)

def test_unrotated_win():
  win_test('none')

def test_rotated_win():
  win_test('single')

def test_arbitrarily_rotated_win():
  win_test('any')

def test_hash():
  # Verify that hash_board and inverse_hash_board are inverses
  keys = randint(1<<16,size=100*4).astype(uint16).view(uint64)
  for i in xrange(len(keys)):
    assert keys[i]==inverse_hash_board(hash_board(keys[i]))
  # Verify that no valid board maps to 0
  try:
    check_board(inverse_hash_board(0))
    assert False
  except ValueError:
    pass

def test_popcounts_over_stabilizers():
  popcounts_over_stabilizers_test(1024)

if __name__=='__main__':
  test_arbitrarily_rotated_win()
