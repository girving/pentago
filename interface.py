#!/usr/bin/env python

from __future__ import division
import os
import re
import sys
from numpy import *
import libpentago as engine
from libpentago import status,pack,unpack,init_table

def flip(board):
  return pack((2*unpack(board))%3)

def flipto(board,turn):
  return flip(board) if turn else board

def show_board(board):
  table = unpack(board)
  return '\n'.join('abcdef'[i]+'  '+''.join('_01'[table[j,5-i]] for j in xrange(6)) for i in xrange(6)) + '\n\n   123456'

def moves(board,turn):
  return [flipto(m,1-turn) for m in engine.moves(flipto(board,turn))]

def lift(score):
  depth,value = divmod(score,4)
  if value!=1:
    assert depth>=36, 'invalid score: depth %d, value %d'%(depth,value)
  return 4*(depth+1)+2-value

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

def move(board,turn,depth,rand=True):
  '''Returns (next,score), where next is the board position moved to and score = depth<<2 | 0 (loss), 1 (tie), or 2 (win).'''
  nexts = engine.moves(flipto(board,turn))
  assert len(nexts)
  options = []
  for next in nexts:
    score = lift(engine.evaluate(depth-1,next))
    assert score//4>=depth, 'unexpected evaluation: depth %d, value %d, expected depth %d'%(score//4,score&3,depth)
    if not options or (options[0][0]&3)<(score&3):
      options = [(score,next)]
      if (score&3)==2:
        break
    elif (options[0][0]&3)==(score&3):
      options.append((score,next))
  score,next = options[random.randint(len(options)) if rand else 0]
  return flipto(next,1-turn),score
 
def play(board,turn,sides,depth,early_exit=False):
  first = turn
  positions = []
  while 1:
    positions.append(board)
    header = 'board = %d, turn = %d'%(board,turn)
    print '\n%s\n%s'%('-'*len(header),header)
    print '\n'+show_board(board)
    s = status(board)
    if s:
      print {1:'win for 0',2:'win for 1',3:'simultaneous win (tie)'}[s]
      value = (s&1)-(s>>1)
      break
    if turn in sides:
      engine.clear_stats()
      next,result = move(board,turn,depth)
      ms = inv_parse_move(board,turn,next)
      d,value = divmod(result,4)
      print "move '%s', depth %d, score %d (%s)"%(ms,d,value-1,('win for '+'01'[turn^(value<1)] if value!=1 else 'tie'))
      print ', '.join('%s = %d'%(k,v) for k,v in engine.stats().items())
      if early_exit and value!=1:
        value = (-1)**turn*(value-1)
        break
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
    board,turn = next,1-turn
  return first,value,positions
