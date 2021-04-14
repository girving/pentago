#!/usr/bin/env python3
"""Test supertensors"""

import boards
import datasets
import jax
import jax.numpy as jnp
import numpy as np
import supertensors as st
import symmetry as sym
import timeit


def test_quads_to_section():
  key = jax.random.PRNGKey(7)
  quads = jax.random.randint(key, (11,4,9), 0, 3)
  s = st.quads_to_section(quads)
  for i in 0, 1:
    assert np.all((quads == 1+i).sum(axis=-1) == s[...,i])
  assert np.all(st.section_sum(s) == (quads != 0).sum(axis=(-1,-2)))


def test_section_sig():
  key = jax.random.PRNGKey(7)
  s = jax.random.randint(key, (11,4,2), 0, 9, dtype=np.uint8)
  assert np.all(st.section_unsig(st.section_sig(s)) == s)


def test_standardize_section():
  key = jax.random.PRNGKey(7)
  s = jax.random.randint(key, (117,4,2), 0, 9, dtype=np.uint8)
  gs, g = st.standardize_section(s) 
  assert np.all(st.section_sig(gs) <= st.section_sig(s))
  assert np.all(gs == sym.transform_section(g, s))


def test_section_misc():
  """Regression test for section functionality which is hard to unit test"""
  s = st.quads_to_section(boards.Board.parse('410395854709526080').quad_grid) 
  assert np.all(s == np.array([[2,1],[1,3],[3,1],[0,1]]))

  # Transformations
  t = sym.transform_section
  assert np.all(t(1, s) == np.array([[1,3],[0,1],[2,1],[3,1]]))
  assert np.all(t(1, t(1, t(1, t(1, s)))) == s)
  assert np.all(t(4, s) == np.array([[0,1],[1,3],[3,1],[2,1]]))
  a, b = jnp.arange(8)[:,None], jnp.arange(8)
  assert np.all(t(a, t(b, s)) == t(sym.global_mul(a, b), s))

  # Standardization
  g = jnp.arange(8)
  ss, h = st.standardize_section(t(g, s)) 
  assert np.all(ss == s)
  assert np.all(sym.global_mul(g, h) == 0)

  # section_sum, section_shape, block_shape
  assert st.section_sum(s) == 12
  shape = [64,126,126,3]
  g = jnp.arange(8)
  assert np.all(st.section_shape(t(g, s)) == sym._transform(g, shape))
  i = jnp.meshgrid(*(jnp.arange(n) for n in (8, 16, 16, 1)), indexing='ij')
  block = jnp.stack(i, axis=-1)
  correct = jnp.stack([8 + 0*i[0], jnp.where(i[1] < 15, 8, 6), jnp.where(i[2] < 15, 8, 6), 3 + 0*i[0]], axis=-1)
  assert np.all(st.block_shape(s, block) == correct)


def test_uninterleave():
  n = 23
  key = jax.random.PRNGKey(7)
  before = jax.random.randint(key, (n, 64), 0, 256).astype(np.uint8)
  after = st.uninterleave(before)
  assert after.dtype == np.uint32
  assert after.shape == (n, 16)
  def bits(x):
    return (x[...,None] >> jnp.arange(8 * x.dtype.itemsize) & 1).astype(np.uint8)
  assert np.all(bits(before).reshape(n,256,2).swapaxes(1,2) == bits(after).reshape(n,2,256))


def test_descendent_sections():
  # Generated from C++ to ensure correct porting
  known = [
    ['00000000'],
    ['10000000'],
    ['11000000', '01100000', '00011000'],
    ['21000000', '11100000', '01200000', '01101000', '10011000', '00111000', '00012000'],
    ['22000000', '12100000', '02200000', '21010000', '11110000', '02101000', '11011000', '01111000', '10021000',
     '00121000', '01012000', '00022000', '20010100', '10110100', '00210100', '00111100', '01011010', '10010110']
  ]
  known_counts = [1,1,3,7,18,31,59,101,177,272,427,631,934,1290,1780,2344,3067,3807,4686]
  known_bits = [0x0,0x1,0x11101,0x31302,0x113102,0x1020123,0x3010211,0x4021220,0x3127061,0x6000000,
                0x55,0x116225,0x320515,0x11205445,0x10334603,0x139546c,0x232e379,0x78e437,0x32238c37]

  def name(s):
    return ''.join(str(k) for k in s.reshape(8))

  # Compare Python and C++
  slices = st.descendent_sections(18)
  assert len(slices) == 18 + 1
  for n, ss in enumerate(slices):
    assert ss.shape == (known_counts[n], 4, 2), f'n {n}'
    assert ss.dtype == np.uint8
    if n < len(known):
      assert known[n] == [st.section_str(s) for s in ss]
    bits = np.bitwise_xor.reduce(st.section_sig(ss))
    assert known_bits[n] == bits, f'n {n}, known {known_bits[n]}, bits {bits}'


def test_supertensors():
  base = '../data/edison/project/all'
  max_slice = 5

  # Load .pentago data
  supers = st.Supertensors(max_slice=max_slice, index_path=base+'-index', super_path=base)
  sections = np.concatenate(supers.sections)
  boards, data = [], []
  for s in sections:
    for I in supers.all_blocks(s):
      b, d = supers.read_block(s, I)
      boards.append(b)
      data.append(d)
  boards = np.concatenate(boards)
  data = np.concatenate(data)
  big = datasets.SuperData(np.concatenate([boards[:,None,:], data.reshape(len(data), 8, 2)], axis=1))

  # Load sparse data
  small = datasets.sparse_dataset(counts=range(max_slice+1))
  if 0:
    print(f'big {len(big)}, small {len(small)}')

  # Verify that small is a subset of big
  def sort(d):
    data = np.asarray(d._data)
    boards = np.ascontiguousarray(data[:,0]).view(np.uint64).squeeze(axis=-1)
    i = np.argsort(boards)
    return boards[i], data[i,1:]
  big_boards, big_supers = sort(big)
  small_boards, small_supers = sort(small)
  i = np.searchsorted(big_boards, small_boards)
  assert np.all(big_boards[i] == small_boards)
  assert np.all(big_supers[i] == small_supers)

  # Test a random sample
  correct = {22799705089704665:0,9831180140546:-1,562988751781888:0,206039682952205052:1,433721649:0,137078314523951104:0,618119744927957073:0,983547510784:-1,22818684502278144:1,56367205646336:-1,57054823317504:1,22800285149036625:0,1848164697092784155:0,24183270:-1,734472044544:0,15199764849819654:1,9517759004672:-1,67491121397760:0,33049773342726:0,1380:1,7484307:0,1851823876074374821:-1,11480601526272:1,4339500411732885504:-1,28183866179584:1,1048258740278:0,1688849865580452:0,661184720293730802:0,205195258022068662:0,1854357842354508057:0,844618299277312:0,1973494522839202:0,22799473448386560:1,820781032378466331:0,3100728343444599645:0}
  datasets.dataset_correctness_test(big, correct=correct, steps=7, batch=5)


if __name__ == '__main__':
  test_supertensors()
