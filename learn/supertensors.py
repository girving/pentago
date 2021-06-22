"""Supertensor reading"""

import aiofiles
from batching import batch_vmap
from functools import partial, cache
import jax
import jax.numpy as jnp
import lzma
import numpy as np
import os
import re
import struct
from symmetry import transform_section
import tables
from trace import scope, scoped

_block_size = 8
_compact_blob_size = 12
_rmq_offsets = jnp.asarray(tables.rotation_minimal_quadrants_offsets)


def quads_to_section(quads):
  assert quads.shape[-2:] == (4,9)
  return (quads[...,None] == 1+jnp.arange(2)).sum(axis=-2).astype(np.uint8)


@partial(jnp.vectorize, signature='(4,2)->()')
def section_sig(section):
  assert section.dtype == np.uint8
  return (section.reshape(8).astype(np.uint32) << 4*jnp.arange(8)).sum().astype(np.uint32)


@partial(jnp.vectorize, signature='()->(4,2)')
def section_unsig(sig):
  assert sig.dtype == np.uint32
  return (sig >> 4*jnp.arange(8) & 15).astype(np.uint8).reshape(4,2)


@partial(jnp.vectorize, signature='(4,2)->(4,2),()')
def standardize_section(section):
  """Returns (g*s, g) s.t. sig(g*s) is minimized"""
  gs = transform_section(jnp.arange(8), section)
  g = jnp.argmin(section_sig(gs))
  return gs[g], g


@partial(jnp.vectorize, signature='(4,2)->()')
def section_sum(section):
  assert section.dtype == np.uint8
  return section.astype(np.int32).sum()


@jax.jit
def section_rmq_i(section):
  assert section.dtype == np.uint8
  black, white = jnp.moveaxis(section, -1, 0).astype(np.int32)
  return ((black * (21-black)) >> 1) + white


@jax.jit
def section_shape(section):
  i = section_rmq_i(section)
  return _rmq_offsets[i + 1] - _rmq_offsets[i]


def section_block_shape(section):
  return (section_shape(section) - 1) // _block_size + 1


def section_str(section):
  return ''.join(str(k) for k in section.reshape(8))


def section_rmqs(section):
  """Rotation minimal quadrants for each quadrant of a section"""
  i = np.asarray(section_rmq_i(section))
  lo = tables.rotation_minimal_quadrants_offsets[i]
  hi = tables.rotation_minimal_quadrants_offsets[i+1]
  return [tables.rotation_minimal_quadrants_flat[l:h] for l, h in zip(lo, hi)]


@jax.jit
def block_shape(section, block):
  return jnp.minimum(_block_size, section_shape(section) - _block_size * block)


def section_all_blocks(section):
  assert section.shape == (4,2), section.shape
  shape = section_block_shape(section)
  def dim(i, s):
    return np.arange(s)[(slice(None),) + (None,)*(3-i)]
  blocks = np.stack([np.broadcast_to(dim(i, s), shape) for i, s in enumerate(shape)], axis=-1)
  return blocks.reshape(shape.prod(), 4)


@batch_vmap(128)
def uninterleave(data):
  """Uninterleave buffer viewed as super_t[...,2]"""
  assert data.dtype == np.uint8
  assert data.shape == (64,)
  a, b = data.reshape(32, 2).T.astype(np.uint32)
  x = a | b << 8
  x = (x | x << 15) & 0x55555555
  x = (x | x >> 1) & 0x33333333
  x = (x | x >> 2) & 0x0f0f0f0f
  x = (x | x >> 4)
  x = jnp.concatenate([x, x >> 16]) & 0xff
  return (x.reshape(16, 4) << 8*jnp.arange(4)).sum(axis=-1).astype(np.uint32)


def descendent_sections(max_slice):
  """Compute all sections that root depends on, organized by slice"""
  assert 0 <= max_slice <= 18
  sentinel = 10 * jnp.ones((4,2), dtype=np.uint8)
  sentinel_sig = section_sig(sentinel)[None]
  unsig = batch_vmap(32)(section_unsig)

  @partial(jnp.vectorize, signature='(4,2)->(4,4,2)')
  def children(s):
    q = jnp.arange(4)[:,None]
    kids = s + (q[:,None] == q) * ((section_sum(s)&1) == jnp.arange(2))
    return jnp.where((kids.sum(axis=-1) <= 9).all(axis=-1)[:,None,None], kids, sentinel)

  @batch_vmap(32)
  def step(s):
    s = children(s)
    s, _ = standardize_section(s)
    return section_sig(s)

  slices = [jnp.zeros((1,4,2), dtype=np.uint8)]
  for n in range(max_slice):
    s = step(slices[-1])
    s = np.unique(np.concatenate([s.reshape(-1), sentinel_sig]))[:-1]
    slices.append(unsig(s))
  return tuple(slices)


class SupertensorIndex:
  def __init__(self, sections):
    self._sigs = section_sig(sections)
    self._shapes = shapes = np.array(section_block_shape(sections), dtype=np.int64)
    def exsum(xs):
      return np.add.accumulate(np.concatenate([[0], xs]))[:-1]
    self._offsets = 24 + _compact_blob_size * exsum(shapes.prod(axis=-1))
    assert self._offsets.dtype == np.int64, self._offsets.dtype

  def blob_location(self, section, I):
    """Where is the compact_blob_t for this block in the index file?"""
    assert I.dtype == np.int64, I.dtype
    s = np.searchsorted(self._sigs, section_sig(section))
    shape = self._shapes[s]
    strides = np.concatenate([np.multiply.accumulate(shape[:0:-1])[::-1], np.ones([1], dtype=np.int64)])
    assert strides.dtype == np.int64
    assert np.all(0 <= I) and np.all(I < shape), f'shape {shape}, I {I}'
    return dict(offset=self._offsets[s] + _compact_blob_size * strides @ I, size=_compact_blob_size)

  @staticmethod
  def block_location(blob):
    """Decode a compact_blob_t"""
    offset, size = struct.unpack('<QL', blob)
    return dict(offset=offset, size=size)


class Supertensors:
  def __init__(self, *, index_path, super_path, max_slice):
    self.sections = sections = descendent_sections(max_slice)
    self._indices = tuple(SupertensorIndex(s) for s in sections)
    self._file = cache(lambda path: open(path, 'rb'))
    self._index = lambda n: f'{index_path}/slice-{n}.pentago.index'
    self._super = lambda n: f'{super_path}/slice-{n}.pentago'

  async def _read(self, path, *, offset, size, client=None):
    if path.startswith('gs://'):
      m = re.match(r'^gs://([^/]+)/(.*)$', path)
      s = await client.download(m.group(1), m.group(2), headers=dict(Range=f'bytes={offset}-{offset+size-1}'))
    else:
      s = os.pread(self._file(path).fileno(), size, offset)
    assert len(s) == size
    return s

  @scoped
  async def read_block(self, section, I, *, client=None):
    assert section.shape == (4,2)
    n = np.asscalar(section_sum(section))
    index = self._indices[n]
    with scope('read'):
      blob = await self._read(self._index(n), **index.blob_location(section, I), client=client)
      data = await self._read(self._super(n), **index.block_location(blob), client=client)
    with scope('process'):
      with scope('decompress'):
        data = lzma.decompress(data)
      with scope('uninterleave'):
        data = np.frombuffer(data, dtype=np.uint8).reshape(-1, 64)
        data = uninterleave(data)
      with scope('boards'):
        shape = tuple(block_shape(section, I))
        assert data.shape == (np.prod(shape, dtype=int), 16)
        q0, q1, q2, q3 = [q[_block_size*i:][:_block_size].astype(np.uint32) for i, q in zip(I,section_rmqs(section))]
        q0 = np.broadcast_to(q0[:,None,None,None], shape)
        q1 = q1[:,None,None]
        q2 = np.broadcast_to(q2[:,None], shape)
        boards = np.stack([q0 | q1 << 16, q2 | q3 << 16], axis=-1)
        boards = boards.reshape(len(data), 2)
      return boards, data
