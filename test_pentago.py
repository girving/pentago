#!/usr/bin/env python

from __future__ import division
from numpy import *
from interface import *

randint = random.randint

def randboard():
  return sum(randint(3**9,size=4)<<16*arange(4)) 

def test_misc():
  random.seed(73120)
  for _ in xrange(100):
    board = randboard()
    assert pack(unpack(board))==board
    assert flip(flip(board))==board
    move = '%s%s %s%s %s'%('abcdef'[randint(6)],'123456'[randint(6)],'ul'[randint(2)],'lr'[randint(2)],'lr'[randint(2)])
    turn = randint(2)
    try:
      next = parse_move(board,turn,move)
      assert next in moves(board,turn)
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

def test_moves():
  assert len(all_moves)==36*8
  random.seed(73120)
  for _ in xrange(2):
    board = randboard()
    turn = randint(2)
    mv = moves(board,turn)
    mv2 = []
    for m in all_moves:
      try:
        mv2.append(parse_move(board,turn,m))
      except ValueError:
        pass
    assert sorted(mv)==sorted(mv2) 
