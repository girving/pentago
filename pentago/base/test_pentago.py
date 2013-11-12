#!/usr/bin/env python

from __future__ import division
from geode import *
from pentago import *
from pentago.misc.interface import *

randint = random.randint

def randboard():
  return sum(randint(3**9,size=4)<<16*arange(4))

def test_misc():
  random.seed(73120)
  for _ in xrange(100):
    board = randboard()
    assert pack(unpack(board))==board
    assert from_table(to_table(board))==board
    assert flip(flip(board))==board
    move = '%s%s %s%s %s'%('abcdef'[randint(6)],'123456'[randint(6)],'ul'[randint(2)],'lr'[randint(2)],'lr'[randint(2)])
    turn = randint(2)
    for simple in False,True:
      if simple:
        move = move[:2]
      try:
        next = parse_move(board,turn,move,simple=simple)
        assert next in moves(board,turn,simple=simple)
      except ValueError:
        pass

def test_known():
  known = [(1484828812733057276,0,'e5 ll l',1484829160625409948),
           (4672209229181495341,1,'a1 ur l',3161251554199873581),
           ( 872030201603702433,1,'f3 ul l',872030202033620051),
           (2060127815417467224,1,'c3 lr r',2060164202475950424)]
  for i,(before,turn,move,after) in enumerate(known):
    if 0:
      print '----------------------------------------'
      print show_board(before)
      print 'turn %d, move: %s'%(turn,move)
      print show_board(after)
    assert parse_move(before,turn,move)==after
    if i==0:
      assert inv_parse_move(before,turn,after)==move

def move_test(simple):
  random.seed(73120)
  for _ in xrange(2):
    board = randboard()
    turn = randint(2)
    mv = moves(board,turn,simple=simple)
    mv2 = []
    for m in all_simple_moves if simple else all_moves:
      try:
        mv2.append(parse_move(board,turn,m,simple=simple))
      except ValueError:
        pass
    assert sorted(mv)==sorted(mv2)

def test_moves():
  assert len(all_moves)==36*8
  move_test(0)

def test_simple_moves():
  assert len(all_simple_moves)==36
  move_test(1)

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
    status = engine.status
    closeness = unrotated_win_closeness
    special = ()
  elif rotations=='single':
    def status(board):
      st = rotated_status(board)
      assert st==(len([() for qx in 0,1 for qy in 0,1 for count in -1,1 if engine.status(rotate(board,qx,qy,count))&1])!=0)
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
        board = choose(moves(board,turn=0,simple=1))
      close = closeness(board)
      dist = distance(close)
      log('seeds')
    while dist:
      # Verify that a well-placed white stone reduces closeness
      try:
        if rotations!='any': # We should almost always be able to find a move reducing closeness
          board,close = choose([(b,c) for b in moves(board,turn=1,simple=1) for c in [closeness(b)] if c<close])
        else: # Black has too much freedom to rotate, so don't require a reduction in closeness
          close,_,board = sorted((closeness(b),randint(1000),b) for b in moves(board,turn=1,simple=1))[0]
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
      board,close,dist = choose([(b,c,d) for b in moves(board,turn=0,simple=1) for c in [closeness(b)] for d in [distance(c)] if d<dist])
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

if __name__=='__main__':
  test_arbitrarily_rotated_win()
