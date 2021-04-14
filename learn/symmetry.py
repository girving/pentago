#!/usr/bin/env python3
"""Pentago symmetries"""

import boards
from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import tables


# Flatten table so that symmetry_mul and symmetry_inv vectorize cleanly
_commute_flat = jnp.asarray(tables.commute_global_local_symmetries.reshape(8 * 256))


def global_mul(a, b):
  # Unit tests to the rescue
  return ((a ^ b) & 4) | (((a ^ (a & b >> 2) << 1) + b) & 3)


def symmetry_mul(a, b):
  def local_mul(a, b):
    return (((a & 0x33) + (b & 0x33)) & 0x33) + (((a & 0xcc) + (b & 0xcc)) & 0xcc)
  ag = a >> 8
  bg = b >> 8
  al = a & 255
  bl = b & 255
  return global_mul(ag, bg) << 8 | local_mul(_commute_flat[bg << 8 | al], bl)


def symmetry_inv(g):
  def local_inv(g):
    return g ^ (g & 0x55) << 1
  def global_inv(g):
    return g ^ (~g >> 2 & (g & 1)) << 1
  li = local_inv(g & 255)
  gi = global_inv(g >> 8)
  return gi << 8 | _commute_flat[gi << 8 | li]


@partial(jnp.vectorize, signature='(),(n)->(n)')
def _transform(g, x):
  n = int(np.sqrt(len(x)))
  x = jax.lax.cond(g & 1, lambda q: q.reshape(n,n)[:,::-1].T.reshape(-1), lambda q: q, x)
  x = jax.lax.cond(g & 2, lambda q: q[::-1], lambda q: q, x)
  x = jax.lax.cond(g & 4, lambda q: q.reshape(n,n)[::-1,::-1].T.reshape(-1), lambda q: q, x)
  return x


@partial(jnp.vectorize, signature='(),(4,9)->(4,9)')
def transform_quads(g, quads):
  """Globally transform board quads by an element of D4"""
  assert g.dtype == np.int32
  assert quads.shape == (4,9)
  transform = partial(_transform, g)
  quads = jax.vmap(transform, 0, 0)(quads)
  quads = jax.vmap(transform, 1, 1)(quads)
  return quads


@partial(jnp.vectorize, signature='(),(4,2)->(4,2)')
def transform_section(g, section):
  """Globally transform a section by an element of D4"""
  assert section.dtype == np.uint8
  return jax.vmap(partial(_transform, g), 1, 1)(section)


@partial(jnp.vectorize, signature='(),(4,9)->(4,9)')
def super_transform_quads(g, quads):
  """Transform board quads according to a supersymmetry"""
  assert g.dtype == np.int32
  quads = jax.vmap(_transform)(g >> 2*jnp.arange(4) & 3, quads)
  return transform_quads(g >> 8, quads)


@partial(jnp.vectorize, signature='(),(2)->(2)')
def super_transform_board(g, board):
  """Transform a board according to a supersymmetry, keeping it in uint64 form"""
  return boards.quads_to_board(super_transform_quads(g, boards.board_to_quads(board)))


@partial(jnp.vectorize, signature='(),(4,w)->(4,w)')
def transform_state(g, x):
  """Globally transform an internal activation state by a rotation g (for now, we ignore reflections)"""
  return jnp.roll(x, g, axis=0)
