#!/usr/bin/env python3
"""Test supertensors"""

import asyncio
import boards
import datasets
import jax
import jax.numpy as jnp
import pytest
import numpy as np
import supertensors as st
import symmetry as sym
import sys
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


@pytest.mark.asyncio
async def test_supertensors(*, local=True):
  if local:
    super_path = '../data/edison/project/all'
    index_path = super_path + '-index'
  else:
    super_path = index_path = 'gs://pentago-us-central1'
  max_slice = 5
  print(f'super_path = {super_path}')

  # Load .pentago data
  supers = st.Supertensors(max_slice=max_slice, index_path=index_path, super_path=super_path)
  sections = np.concatenate(supers.sections)
  boards, data = [], []
  for s in sections:
    for I in st.section_all_blocks(s):
      b, d = await supers.read_block(s, I)
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
  correct = {3247887155200:1,695887331571:-1,3898709902419297147:1,106397541196635987:0,479528532:-1,149463212633358339:1,45598946227127166:1,1846766715435422136:0,48103595:0,22799821011615906:-1,273595764716863575:0,4480827392:0,5171249233:0,1044537147644:-1,615609778643730432:0,683984270716305897:0,236653314075:0,59490117615616:1,685673043410489622:-1,136797882788414223:1,3716595592487501907:-1,205758208073400320:0,1846776108396184362:1,45600337797120162:0,615867249138467551:1,136796839403388928:0,461832193:0,3830311483223703633:0,9667974922240:0,9402543439873:0,2534674981584896:0,7305914548953:-1,6386616565760:1,348039413766:0,22855840269664337:-1}
  datasets.dataset_correctness_test(big, correct=correct, steps=7, batch=5)


if __name__ == '__main__':
  asyncio.run(test_supertensors(local='--gs' not in sys.argv))
