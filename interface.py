#!/usr/bin/env python

from __future__ import division
import os
import re
import sys
from numpy import *
import engine
from engine import status

def unpack(board):
  table = zeros((6,6),dtype=int)
  for x in xrange(6):
    for y in xrange(6):
      table[x,y] = (board>>16*(2*(x//3)+y//3)&0xffff)//3**(3*(x%3)+y%3)%3
  return table

def pack(table):
  board = 0
  for x in xrange(6):
    for y in xrange(6):
      board += table[x,y]*3**(3*(x%3)+y%3)<<16*(2*(x//3)+y//3)
  return board

def flip(board):
  return pack((2*unpack(board))%3)

def flipto(board,turn):
  return flip(board) if turn else board

def show_board(board):
  table = unpack(board)
  return '\n'.join('abcdef'[i]+'  '+''.join('_01'[table[j,5-i]] for j in xrange(6)) for i in xrange(6)) + '\n\n   123456'

def moves(board,turn):
  return [flipto(m,1-turn) for m in engine.moves(flipto(board,turn))]

move_pattern = re.compile(r'^\s*([abcdef])([123456])\s*([ul])([lr])\s*([lr])\s*$')
def parse_move(board,turn,move):
  m = move_pattern.match(move)
  if not m:
    raise SyntaxError("syntax error in move string '%s', example syntax: 'a2 ur l' for 'a2, upper right, rotate left'"%move.strip())
  x = '123456'.find(m.group(2))
  y = 'fedcba'.find(m.group(1))
  qx = 3*'lr'.find(m.group(4))
  qy = 3*'lu'.find(m.group(3))
  dir = m.group(5)
  table = unpack(board)
  if table[x,y]:
    raise ValueError("illegal move '%s', position is already occupied"%move.strip())
  table[x,y] = 1+turn
  for i in xrange({'l':1,'r':3}[dir]):
    copy = table.copy()
    for x in xrange(3):
      for y in xrange(3):
        copy[qx+x,qy+y] = table[qx+y,qy+2-x]
    table = copy
  return pack(table)

all_moves = ['%s%s %s%s %s'%(i,j,qi,qj,d) for i in 'abcdef' for j in '123456' for qi in 'ul' for qj in 'lr' for d in 'lr']
def inv_parse_move(board,turn,next):
  for m in all_moves:
    try:
      if parse_move(board,turn,m)==next:
        return m
    except ValueError:
      pass
  raise ValueError('invalid move')

def move(board,turn,depth):
  '''Returns (next,result), where next is the board position moved to and result = -1 (loss), 0 (tie), or 1 (win).'''
  nexts = engine.moves(flipto(board,turn))
  assert nexts
  best_result = -2
  for next in nexts:
    result = -engine.evaluate(depth,next)
    if best_result<result:
      best_result = result
      best_next = next
      if result==1:
        break
  return flipto(best_next,1-turn),best_result
 
def play(board,turn,sides,depth):
  print '\n---------------------------------------------------------'
  print 'board = %d, turn = %d\n\n%s'%(board,turn,show_board(board))
  s = status(board)
  if s:
    print {1:'win for 0',2:'win for 1',3:'simultaneous win (tie)'}[s]
    return
  if turn in sides:
    next,result = move(board,turn,depth)
    ms = inv_parse_move(board,turn,next)
    print "move '%s', score %d (%s)"%(ms,result,('win for '+'01'[turn^(result<0)] if result else 'tie'))
  else:
    while 1:
      ms = raw_input('enter move (example: a2 ur l): ')
      try:
        next = parse_move(board,turn,ms) 
      except ValueError:
        continue
      except SyntaxError:
        continue
      break
  play(next,1-turn,sides,depth)
