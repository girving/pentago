'''
Pentago lookup table precomputation:

In order to avoid needless work, a good chunk of the logic for basic pentago
operations is baked into lookup tables computed by this script.  This includes
state transformation (rotations, base-2 to/from base-3, etc.), win computation,
symmetry helpers, etc.  These tables also show up in the file formats, since
the ordering of boards within a section is defined by rotation_minimal_quadrants.
'''

from __future__ import division, print_function
import os
import sys
import hashlib
import numpy as np

def cache(f):
  """Cache a nullary function call."""
  value = []
  def cached():
    if not value:
      value.append(f())
    return value[0]
  return cached

remembered = []
def remember(f):
  """Decorate a table we want to save."""
  f = cache(f)
  remembered.append(f)
  return f

def qbit(x,y):
  return 1<<(3*x+y)

def popcount(x):
  count = 0
  while x:
    x &= ~(x&-x)
    count += 1
  return count

@cache
def small_popcounts():
  return np.asarray([popcount(i) for i in range(512)])

def bits(x):
  if not x:
    return '0b0'
  s = ''
  while x:
    s = '01'[x&1]+s
    x //= 2
  return '0b'+s

def check(*args):
  '''Usage: check(arrays,expected)'''
  expected = args[-1]
  data = b''.join(a.astype(a.dtype.newbyteorder('<')).tostring() for a in args[:-1])
  hash = hashlib.sha1(data).hexdigest()
  assert expected==hash, "hash mismatch: expected '%s', got '%s'"%(expected,hash)

@cache
def win_patterns():
  def bit(x,y,flip=False,rotate=False):
    assert 0<=x<6
    assert 0<=y<6
    if flip:
      x,y = y,x
    if rotate:
      x,y = 5-y,x
    return 1<<(16*(2*(x//3)+y//3)+3*(x%3)+y%3)

  # List the various ways of winning
  wins = []
  # Off-centered diagonal 5 in a row
  for x in 0,1:
    for y in 0,1:
      p = 0
      for i in -2,-1,0,1,2:
        p |= bit(2+x+i,2+y+i*(-1 if x==y else 1))
      wins.append(p)
  # Axis aligned 5 in a row
  for flip in 0,1:
    for x in 0,1:
      for y in range(6):
        p = 0
        for i in range(5):
          p |= bit(x+i,y,flip=flip)
        wins.append(p)
  # Centered diagonal 5 in a row
  for rotate in 0,1:
    for x in 0,1:
      p = 0
      for i in range(5):
        p |= bit(i+x,i+x,rotate=rotate)
      wins.append(p)
  wins = np.asarray(wins)

  # Check that 5 bits are set in each
  for w in wins:
    assert popcount(w)==5

  # There should be 4*3*2+4+4 == 32 ways to win.
  # The first four of these are special: they require contributions from three quadrants.
  # The remaining 28 require contributions from only two quadrants.
  assert len(wins)==32==4*3*2+4+4
  return wins

@cache
def win_contributions():
  wins = win_patterns()
  # For each possible value of each quadrant, determine all the ways that quadrant can
  # contribute to victory.
  table = np.zeros((4,512),dtype=np.int64)
  for qx in range(2):
    for qy in range(2):
      q = 2*qx+qy
      qb = 0x1ff<<16*q
      for v in range(512):
        b = v<<16*q
        for i,w in enumerate(wins):
          if w&qb and not ~b&(w&qb):
            table[q,v] |= 1<<(2*i)
  check(table,'4e5cf35e82fceecd464d73c3de35e6af4f75ee34')
  return table

@remember
def show_win_contributions():
  return [('uint64_t','win_contributions','0x%xL',win_contributions())]

@cache
def rotations():
  table = np.zeros((512,2),dtype=np.int16)
  for v in range(512):
    left = 0
    right = 0
    for x in range(3):
      for y in range(3):
        if v&qbit(x,y):
          left |= qbit(2-y,x)
          right |= qbit(y,2-x)
    assert popcount(v)==popcount(left)==popcount(right)
    table[v] = left,right
  check(table,'195f19d49311f82139a18ae681592de02b9954bc')
  return table

@remember
def show_rotations():
  return [('uint16_t','rotations','0x%x',rotations())]

@remember
def rotated_win_contributions():
  wins = win_contributions()
  rot = rotations()
  rwins = wins|wins[:,rot[:,0]]|wins[:,rot[:,1]]
  deltas = rwins - wins
  return [('uint64_t','rotated_win_contribution_deltas','0x%xL',deltas)]

def if_(c,a,b):
  c = c!=0
  return c*a+(1-c)*b

@remember
def unrotated_win_distances():
  # Work out the various ways of winning
  patterns = win_patterns().reshape(2,16)

  # Precompute small popcounts
  count = small_popcounts()

  # Generate a distance table giving the total number of moves required for player 0 (black) to
  # reach the pattern, or 4 for unreachable (a white stone is in the way).  Each pattern distance
  # has 3 bits (for 0,1,2,3 or 4=infinity) plus an extra 0 bit to stop carries at runtime, so we
  # fit 16 patterns into each of 2 64-bit ints
  table = np.zeros((4,3**9,2),np.int64) # indexed by quadrant,position,word
  unpack_ = unpack()
  for i,pats in enumerate(patterns):
    for j,pat in enumerate(pats):
      for q in range(4):
        b = np.arange(3**9)
        s0,s1 = unpack_[b].T
        qp = (pat>>16*q)&0x1ff
        d = if_(s1&qp,4,count[qp&~s0])
        table[q,b,i] |= d<<4*j
  check(table,'02b780e3172e11b861dd3106fc068ccb59cebc1c')
  return [('uint64_t','unrotated_win_distances','0x%xL',table)]

@remember
def arbitrarily_rotated_win_distances():
  # Work out the various ways of winning, allowing arbitrarily many rotations
  patterns = win_patterns().reshape(2,16)

  # Precompute table of multistep rotations
  rotate = np.empty((512,4),np.int16)
  rotate[:,0] = np.arange(512)
  for i in range(3):
    rotate[:,i+1] = rotations()[rotate[:,i],0]

  # Precompute small popcounts
  count = small_popcounts()

  # Generate a distance table giving the total number of moves required for player 0 (black) to
  # reach the pattern, or 4 for unreachable (a white stone is in the way).  Each pattern distance
  # has 3 bits (for 0,1,2,3 or 4=infinity) plus an extra 0 bit to stop carries at runtime, so we
  # fit 16 patterns into each of 2 64-bit ints
  table = np.zeros((4,3**9,2),np.int64) # indexed by quadrant,position,word
  unpack_ = unpack()
  for i,pats in enumerate(patterns):
    for j,pat in enumerate(pats):
      for q in range(4):
        b = np.arange(3**9)
        s0,s1 = unpack_[b].T
        s0 = rotate[s0,:]
        s1 = rotate[s1,:]
        qp = (pat>>16*q)&0x1ff
        d = (if_(s1&qp,4,count[qp&~s0])).min(axis=-1)
        table[q,b,i] |= d<<4*j
  check(table,'b1bd000ba42513ee696f065503d68f62b98ac85e')
  return [('uint64_t','arbitrarily_rotated_win_distances','0x%xL',table)]

@remember
def rotated_win_distances():
  # Work out the various ways of winning, allowing at most one quadrant to rotate
  patterns = win_patterns()
  assert len(patterns)==32
  rotated_patterns = [] # List of pattern,rotated_quadrant pairs
  for pat in patterns:
    for q in range(4):
      if pat&(0x1ff<<16*q):
        rotated_patterns.append((pat,q))
  assert len(rotated_patterns)==68
  rotated_patterns = np.asarray(rotated_patterns,object).reshape(4,17,2)

  # Precompute small popcounts
  count = small_popcounts()

  # Generate a distance table giving the total number of moves required for player 0 (black) to
  # reach the pattern, or 4 for unreachable (a white stone is in the way).  Each pattern distance
  # has 3 bits (for 0,1,2,3 or 4=infinity), so we fit 17 patterns into each of 4 64-bit ints
  table = np.zeros((4,3**9,4),np.int64) # indexed by quadrant,position,word
  unpack_ = unpack()
  rotations_ = rotations()
  for i,pats in enumerate(rotated_patterns):
    for j,(pat,qr) in enumerate(pats):
      for q in range(4):
        b = np.arange(3**9)
        s0,s1 = unpack_[b].T
        qp = (pat>>16*q)&0x1ff
        d = if_(s1&qp,4,count[qp&~s0])
        if q==qr:
          for r in 0,1:
            d = np.minimum(d,if_(rotations_[s1,r]&qp,4,count[qp&~rotations_[s0,r]]))
        table[q,b,i] |= d<<3*j
  check(table,'6fc4ae84c574d330f38e3f07b37ece103fa80c45')
  return [('uint64_t','rotated_win_distances','0x%xL',table)]

@cache
def reflections():
  table = np.zeros(512,dtype=np.int16)
  for v in range(512):
    r = 0
    for x in range(3):
      for y in range(3):
        if v&qbit(x,y):
          r |= qbit(2-y,2-x) # Reflect about x = y line
    assert popcount(v)==popcount(r)
    table[v] = r
  check(table,'2b23dc37f4bc1008eba3df0ee1b7815675b658bf')
  return table

@remember
def show_reflections():
  return [('uint16_t','reflections','0x%x',reflections())]

'''There are 3**9 = 19683 possible states in each quadrant.  3**9 < 2**16, so we can store
a quadrant state in 16 bits using radix 3.  However, radix 3 is inconvenient for computation,
so we need lookup tables to go to and from packed form.'''
@cache
def pack():
  pack = np.zeros(2**9,dtype=np.int16)
  for v in range(512):
    pv = 0
    for i in range(9):
      if v&2**i:
        pv += 3**i
    pack[v] = pv
  check(pack,'b86e92ca7f525bd398ba376616219831e3f4f1a5')
  return pack

@remember
def show_pack():
  return [('uint16_t','pack_table','%d',pack())]

@cache
def unpack():
  unpack = np.zeros((3**9,2),dtype=np.int16)
  for v in range(3**9):
    vv = v
    p0,p1 = 0,0
    for i in range(9):
      c = vv%3
      if c==1:
        p0 += 2**i
      elif c==2:
        p1 += 2**i
      vv //= 3
    unpack[v] = p0,p1
  check(unpack,'99e742106ae60b62c0bb71dee15789ef1eb761a0')
  return unpack

@remember
def show_unpack():
  return [('uint16_t','unpack_table','0x%x',unpack())]

# pack and unpack should be inverses
assert np.all(pack()[unpack()[:,0]]+2*pack()[unpack()[:,1]]==np.arange(3**9))

@remember
def moves():
  '''Given a quadrant, compute all possible moves by player 0 as a function of the set of filled spaces, stored as a nested array.'''
  moves = []
  for filled in range(2**9):
    mv = []
    free = ~filled
    for i in range(9):
      b = 1<<i
      if free&b:
        mv.append(b)
    assert len(mv)==9-popcount(filled)
    moves.append(mv)
  # Pack into a flat nested array
  sizes = np.asarray(list(map(len,moves)))
  offsets = np.hstack([0, np.cumsum(sizes)])
  flat = np.asarray([x for mv in moves for x in mv])
  assert len(flat)==offsets[-1], 'len(flat) = %s != offsets[-1] = %s' % (len(flat), offsets[-1])
  assert len(flat)<2**16
  check(offsets,'15b71d6e563787b098860ae0afb1d1aede6e91c2')
  check(flat,'1aef3a03571fe13f0ab71173d79da107c65436e0')
  return [('uint16_t','move_offsets','%d',offsets)
         ,('uint16_t','move_flat','%d',flat)]

@remember
def superwin_info():
  all_rotations = np.empty((512,4),np.int16)
  all_rotations[:,0] = np.arange(512)
  for i in range(3):
    all_rotations[:,i+1] = rotations()[all_rotations[:,i],0]
  ways = win_patterns().view(np.int16).reshape(32,4)
  if sys.byteorder=='big': # Fix order if necessary
    ways = ways[:,::-1]
  types = dict(map(reversed,enumerate('h v dl dh da'.split()))) # horizontal, vertical, diagonal lo/hi/assist
  patterns = 'v v - - | h - h - | - h - h | - - v v | dl - da dl | dh da - dh | da dl dl - | - dh dh da'
  patterns = [p.split() for p in patterns.split('|')]
  info = np.zeros((4,512,5,4,4,4,4),bool) # Indexed by quadrants, quadrant state, superwin_info field, r0, r1, r2, r3
  for pattern in patterns:
    debug = False # pattern=='- dh dh da'.split()
    if debug: b = [0,136,50,64]
    assert len(pattern)==4
    used   = [i for i in range(4) if pattern[i]!='-']
    unused = [i for i in range(4) if pattern[i]=='-']
    assert len(used) in (2,3)
    ws = np.asarray([w for w in ways if np.all([bool(w[i])==(pattern[i]!='-') or pattern[i]=='da' for i in range(4)])])
    assert len(ws) in (6,3)
    assert len(ws)<=4**(4-len(used))
    for q in used:
      for i,w in enumerate(ws[:,q]):
        s = info[q,:,types[pattern[q]]][:,None,None,None,None,...].swapaxes(4,5+q)
        for j,u in enumerate(unused):
          s = s.swapaxes(1+j,5+u)
        x = (all_rotations&w)==w
        if debug:
          print('pattern %s, q %d, i %d, w %d (%s), x[%d] %s' % (
                pattern,q,i,w,' '.join(str(i) for i in range(9) if w&1<<i),b[q],x[b[q]]))
          before = info.copy()
        if len(unused)==1:
          s[:,i] |= x[:,None,None,:,None,None,None,None]
        else:
          s[:,i//4,i%4] |= x[:,None,:,None,None,None,None]
        if 0 and debug:
          changed = zip(*np.nonzero(info[q,b[q]]!=before[q,b[q]]))
          print('  changed = %d'%len(changed))
          for c in changed:
            print('  %s'%(c,))
  # Switch to r3, r2, r1, r0 major order to match super_t
  info = info.astype(np.int8).swapaxes(-4,-1).swapaxes(-3,-2)
  # packbits uses big endian bit order (on all platforms), so flip around to little endian before packing
  info = np.packbits(info.reshape(-1,8)[:,::-1])
  # Check hash before converting to final endianness
  check(info,'668eb0a940489f434f804d994698a4fc85f5b576')
  # Account for big endianness if necessary.  Note that the byte order for the entire 256 bit super_t is reversed
  if sys.byteorder=='big':
    info = info.reshape(-1,256//8)[:,::-1]
  # Interpret as 64 bit chunks
  info = np.ascontiguousarray(info).view(np.uint64)
  return [('uint64_t','superwin_info','0x%xL',info)]

@remember
def commute_global_local_symmetries():
  # Represent the various symmetries as subgroups of S_{6*6}
  n = np.arange(6)
  identity = np.empty((6,6,2),np.int8)
  identity[:,:,0] = n.reshape(-1,1)
  identity[:,:,1] = n.reshape(1,-1)
  reflect = identity[n[::-1].reshape(1,-1),n[::-1].reshape(-1,1)]
  assert np.all(reflect[0,0]==[5,5])
  assert np.all(reflect[5,0]==[5,0])
  gr = identity[n[::-1].reshape(1,-1),n.reshape(-1,1)]
  assert np.all(gr[0,0]==[5,0])
  assert np.all(gr[5,0]==[5,5])
  lr = np.asarray([identity.copy() for q in range(4)])
  n = np.arange(3)
  for qx in 0,1:
    for qy in 0,1:
      q = 2*qx+qy
      lq = lr[q,3*qx:,3*qy:][:3,:3]
      lq[:] = lq[n[::-1].reshape(1,-1),n.reshape(-1,1)]
      assert np.all(lr[q,3*qx,3*qy]==[3*qx+2,3*qy])
      assert np.all(lr[q,3*qx+2,3*qy]==[3*qx+2,3*qy+2])

  # Flattten everything from S_{6*6} to S_36
  def flatten(p):
    return (6*p[...,0]+p[...,1]).reshape(*(p.shape[:-3]+(36,)))
  identity = flatten(identity)
  reflect = flatten(reflect)
  gr = flatten(gr)
  lr = flatten(lr)
  for f in identity,reflect:
    assert np.all(f[f]==identity)
  for r in (gr,)+tuple(lr):
    assert np.all(r[r[r[r]]]==identity)

  # Construct local rotation group
  def powers(g):
    p = np.empty((4,36),int)
    p[0] = identity
    for i in range(3):
      p[i+1] = g[p[i]]
    return p
  plr = np.asarray(list(map(powers,lr)))
  local = np.empty((4,4,4,4,36),int)
  for i0 in range(4):
    for i1 in range(4):
      for i2 in range(4):
        for i3 in range(4):
          local[i0,i1,i2,i3] = plr[0,i0,plr[1,i1,plr[2,i2,plr[3,i3]]]]
  local = local.reshape(256,36)
  # Verify commutativity
  prod = local[:,local]
  assert np.all(prod==prod.swapaxes(0,1))

  # Construct global rotation group
  globe = np.empty((2,4,36),int)
  globe[0] = powers(gr)
  for i in range(4):
    globe[1,i] = reflect[globe[0,i]]
  globe = globe.reshape(8,36)
  inv_globe = np.empty_like(globe)
  for i in range(8):
    inv_globe[i,globe[i]] = identity

  # Compute conjugations
  def inv(p):
    ip = np.empty_like(p)
    ip[p] = identity
    return ip
  i = np.arange(8).reshape(-1,1)
  j = np.arange(256).reshape(1,-1)
  conjugations = np.asarray([inv(g)[local[:,g]] for g in globe])

  # Generate lookup table
  table = np.all(conjugations==local.reshape(-1,1,1,36),axis=-1).argmax(axis=0)
  assert table.shape==(8,256)
  check(table,'e051d034c07bfa79fa62273b05839aedf446d499')
  return [('uint8_t','commute_global_local_symmetries','%d',table)]

@remember
def superstandardize_table():
  # Given the relative rankings of the four quadrants, determine the global rotation that minimizes the board value
  rotate = np.empty((4,4),int)
  rotate[0] = np.arange(4)
  rotate[1] = [2,0,3,1]
  rotate[2] = rotate[1,rotate[1]]
  rotate[3] = rotate[2,rotate[1]]
  def v(i):
    shape = np.ones(5,int)
    shape[i] = 4
    return np.arange(4).reshape(*shape)
  table = sum([v(i)*4**rotate[v(4),i] for i in range(4)]).argmin(axis=-1).T.ravel()
  check(table,'dd4f59fea3135a860e76ed397b8f1863b23cc17b')
  return [('uint8_t','superstandardize_table','%d',table)]

@remember
def rotation_minimal_quadrants():
  pack_ = pack()
  unpack_ = unpack()
  reflect = reflections()
  rotate = rotations()[:,0]

  # Find quadrants minimal w.r.t. rotations but not necessarily reflections
  minq = np.arange(3**9)
  s0,s1 = unpack_.T
  for r in 1,2,3:
    s0 = rotate[s0]
    s1 = rotate[s1]
    minq = np.minimum(minq,pack_[s0]+2*pack_[s1])
  all_rmins, = np.nonzero(minq==np.arange(3**9))

  # Sort quadrants so that reflected versions are consecutive (after rotation minimizing), and all pairs come first
  s0,s1 = unpack_.T
  reflected = minq[pack_[reflect[s0]]+2*pack_[reflect[s1]]]
  all_rmins = all_rmins[np.argsort(3**9*(all_rmins==reflected[all_rmins])+np.minimum(all_rmins,reflected[all_rmins]),kind='mergesort')]
  def ordered(qs):
    return np.all(reflected[qs]==qs[np.arange(len(qs))^(qs!=reflected[qs])])
  assert ordered(all_rmins)

  # Partition quadrants by stone counts
  s0,s1 = unpack_[all_rmins].T
  b = small_popcounts()[s0]
  w = small_popcounts()[s1]
  i = ((b*(21-b))//2)+w
  rmins = [all_rmins[np.nonzero(i==k)[0]] for k in range(10*(10+1)//2)]
  assert sum(map(len,rmins))==len(all_rmins)

  # Count the number of elements in each bucket not fixed by reflection
  moved = np.asarray([sum(reflected[r]!=r) for r in rmins])
  assert np.all((moved&1)==0)
  for r in rmins:
    assert ordered(r)

  # Compute inverse.  inverse[q] = 4*i+r if rmins[?][i] rotated left 90*r degrees is q
  inverse = np.empty(3**9,int)
  inverse[:] = 4*3**9
  for r in rmins:
    s0,s1 = unpack_[r].T
    for i in range(4):
      r = pack_[s0]+2*pack_[s1]
      inverse[r] = np.minimum(inverse[r],4*np.arange(len(r))+i)
      s0 = rotate[s0]
      s1 = rotate[s1]
  assert np.all(inverse<420*4)

  # Save as a nested array
  offsets = np.cumsum(np.hstack([0,list(map(len,rmins))]))
  flat = np.hstack(rmins)
  check(offsets,'7e450e73e0d54bd3591710e10f4aa76dbcbbd715')
  check(flat,'8f48bb94ad675de569b07cca98a2e930b06b45ac')
  check(inverse,'339369694f78d4a197db8dc41a1f41300ba4f46c')
  check(moved,'dce212878aaebbcd995a8a0308335972bd1d5ef7')
  return [('uint16_t','rotation_minimal_quadrants_offsets','%d',offsets)
         ,('uint16_t','rotation_minimal_quadrants_flat','%d',np.hstack(rmins))
         ,('uint16_t','rotation_minimal_quadrants_inverse','%d',inverse)
         ,('uint16_t','rotation_minimal_quadrants_reflect_moved','%d',moved)]

def cpp_sizes(data):
  return ''.join('[%d]'%n for n in data.shape)

def cpp_init(format,data):
  if data.shape:
    return '{'+','.join(cpp_init(format,d) for d in data)+'}'
  else:
    return format%data

def save(h, cc, tables):
  note = '// Autogenerated by precompute: do not edit directly\n'
  for path in h, cc:
    file = open(path, 'w')
    print(note, file=file)
    if path.endswith('.h'):
      print('#pragma once\n\n#include <cstdint>', file=file)
    else:
      print('#include "pentago/base/gen/tables.h"', file=file)
    print('namespace pentago {\n', file=file)
    for type,name,format,data in tables:
      if path.endswith('.h'):
        print('extern const %s %s%s;'%(type,name,cpp_sizes(data)), file=file)
      else:
        print('const %s %s%s = %s;'%(type,name,cpp_sizes(data),cpp_init(format,data)), file=file)
    print('\n}', file=file)

def main():
  h, cc = sys.argv[1:]
  assert h.endswith('tables.h')
  assert cc.endswith('tables.cc')
  save(h, cc, [t for f in remembered for t in f()])

if __name__ == '__main__':
  main()
