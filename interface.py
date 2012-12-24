#!/usr/bin/env python

from __future__ import division
import os
import re
import sys
from numpy import *
import pentago_core as engine
from pentago_core import *

def is_super(board):
  try:
    board,rotation = board
    return True
  except:
    return False

def flip(board):
  if is_super(board):
    board,rotation = board 
    return flip(board),rotation
  return pack(ascontiguousarray(unpack(board)[::-1]))

def flipto(board,turn):
  return flip(board) if turn else board

def reduce_board(board):
  return rotate(*board) if is_super(board) else board

def show_board(board,brief=False):
  if is_super(board):
    if brief:
      board,rotation = board
      return '%d,%d,%d,%d,%d'%((board,)+tuple(rotation))
    else:
      return show_board(rotate(*board))
  elif brief:
    return str(board)
  else:
    table = to_table(board)
    return '\n'.join('abcdef'[i]+'  '+''.join('_01'[table[j,5-i]] for j in xrange(6)) for i in xrange(6)) + '\n\n   123456'

def show_side(side):
  return '\n'.join('abcdef'[5-y]+'  '+''.join('_0'[bool(side&1<<16*(x//3*2+y//3)+x%3*3+y%3)] for x in xrange(6)) for y in reversed(xrange(6))) + '\n\n   123456'

def moves(board,turn,simple=False):
  return [flipto(m,1-turn) for m in (engine.simple_moves if simple else engine.moves)(flipto(board,turn))]

def lift(score):
  depth,value = divmod(score,4)
  assert depth>=0 and 0<=value<=2, 'invalid score: depth %d, value %d'%(depth,value)
  return 4*(depth+1)+2-value

def rotate(board,*args):
  '''Usage: either rotate(board,qx,qy,count) or rotate(board,tuple)'''
  if len(args)==1:
    for x in 0,1:
      for y in 0,1:
        board = rotate(board,x,y,args[0][2*x+y])
    return board
  else:
    qx,qy,count = args
    count = (count%4+4)%4
    table = to_table(board)
    for i in xrange(count):
      copy = table.copy()
      for x in xrange(3):
        for y in xrange(3):
          copy[3*qx+x,3*qy+y] = table[3*qx+y,3*qy+2-x]
      table = copy
    return from_table(table)

move_pattern = re.compile(r'^\s*([abcdef])([123456])\s*([ul])([lr])\s*([lr])\s*$')
simple_move_pattern = re.compile(r'^\s*([abcdef])([123456])\s*$')
def parse_move(board,turn,move,simple=False):
  m = (simple_move_pattern if simple else move_pattern).match(move)
  if not m:
    if simple:
      raise SyntaxError("syntax error in move string '%s', example syntax: 'a2'"%move.strip())
    else:
      raise SyntaxError("syntax error in move string '%s', example syntax: 'a2 ur l' for 'a2, upper right, rotate left'"%move.strip())
  x = '123456'.find(m.group(2))
  y = 'fedcba'.find(m.group(1))
  if not simple:
    qx = 'lr'.find(m.group(4))
    qy = 'lu'.find(m.group(3))
    dir = m.group(5)
  table = to_table(board)
  if table[x,y]:
    raise ValueError("illegal move '%s', position is already occupied"%move.strip())
  table[x,y] = 1+turn
  if simple:
    return from_table(table)
  else:
    return rotate(from_table(table),qx,qy,{'l':1,'r':-1}[dir])

all_moves = ['%s%s %s%s %s'%(i,j,qi,qj,d) for i in 'abcdef' for j in '123456' for qi in 'ul' for qj in 'lr' for d in 'lr']
all_simple_moves = ['%s%s'%(i,j) for i in 'abcdef' for j in '123456']
def inv_parse_move(board,turn,next,simple=False):
  board = reduce_board(board)
  next = reduce_board(next)
  for m in all_simple_moves if simple else all_moves:
    try:
      if parse_move(board,turn,m,simple)==next:
        return m
    except ValueError:
      pass
  raise ValueError('invalid move')

def move(board,turn,depth,rand=True,simple=False):
  '''Returns (next,score), where next is the board position moved to and score = depth<<2 | 0 (loss), 1 (tie), or 2 (win).'''
  if is_super(board):
    flipped = flipto(board,turn)
    results = engine.super_evaluate_children(black_to_move(flipped[0]),depth,*flipped)
    nexts = [r[0] for r in results]
    results = dict(results)
  else:
    nexts = (engine.simple_moves if simple else engine.moves)(flipto(board,turn))
  assert len(nexts)
  options = []
  for next in nexts:
    score = results[next] if is_super(board) else lift((engine.simple_evaluate if simple else engine.evaluate)(depth-1,next))
    assert score//4>=depth, 'unexpected evaluation: depth %d, value %d, expected depth %d'%(score//4,score&3,depth)
    if not options or (options[0][0]&3)<(score&3):
      options = [(score,next)]
      if (score&3)>=(1 if simple and turn==1 else 2):
        break
    elif (options[0][0]&3)==(score&3):
      options.append((score,next))
  score,next = options[random.randint(len(options)) if rand else 0]
  return flipto(next,1-turn),score

def play(board,turn,sides,depth,early_exit=False,simple=False,brief=36):
  assert brief
  first = turn
  positions = []
  final_result = None
  while 1:
    positions.append(int(reduce_board(board)))
    header = 'board = %s, turn = %d'%(show_board(reduce_board(board),brief=1),turn)
    print '\n%s\n%s'%('-'*len(header),header)
    print '\n'+show_board(board)
    s = (rotated_status if simple else status)(reduce_board(board))
    if s:
      print {1:'win for 0',2:'win for 1',3:'simultaneous win (tie)'}[s]
      if simple and not status(board):
        for qd in xrange(8):
          qx = qd//4
          qy = qd//2&1
          c = (-1)**(qd&1)
          rb = rotate(board,qx,qy,c)
          if status(rb)==1:
            print 'rotate %s%s %s to get'%('lu'[qy],'lr'[qx],('left','right')[c<0])
            print show_board(rb)
            break
        else:
          raise RuntimeError('not a rotated win for black')
      value = (s&1)-(s>>1)
      break
    if not brief:
      break 
    brief -= 1
    if turn in sides:
      engine.clear_stats()
      print 'search depth %d'%depth
      next,result = move(board,turn,depth,simple=simple)
      ms = inv_parse_move(board,turn,next,simple=simple)
      d,value = divmod(result,4)
      print "move '%s', depth %d, score %d (%s)"%(ms,d,value-1,('win for '+'01'[turn^(value<1)] if value!=1 else 'tie'))
      engine.print_stats()
      print 'total moves = %d'%sum(to_table(reduce_board(next))!=0)
      fr = (-1)**turn*(value-1)
      if final_result is not None:
        assert final_result==fr
      elif d>=36:
        final_result = fr
      if early_exit and value!=1:
        value = (-1)**turn*(value-1)
        break
    else:
      while 1:
        ms = raw_input('enter move (example: %s): '%('a2' if simple else 'a2 ur l'))
        try:
          next = parse_move(board,turn,ms,simple)
        except ValueError:
          continue
        except SyntaxError:
          continue
        break
    board,turn = next,1-turn
    # Reduce search depth so that the game ends quickly
    if final_result is not None:
      depth -= 1
  return first,value,positions

def factorial(n):
  assert n>=0
  f = 1
  for i in xrange(2,n+1):
    f *= i
  return f

def choose(n,k):
  if k<0 or k>n: return 0
  return factorial(n)//(factorial(k)*factorial(n-k))
