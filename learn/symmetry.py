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


def symmetry_mul(a, b):
  def local_mul(a, b):
    return (((a & 0x33) + (b & 0x33)) & 0x33) + (((a & 0xcc) + (b & 0xcc)) & 0xcc)
  ag = a >> 8
  bg = b >> 8
  al = a & 255
  bl = b & 255
  # We seek x = ag al bg bl
  # Commute local and global transforms: al bg = bg (bg' al bg) = bg l2, x = ag bg l2 bl
  l2 = _commute_flat[bg << 8 | al]
  # Unit tests to the rescue
  xg = ((ag ^ bg) & 4) | (((ag ^ (ag & bg >> 2) << 1) + bg) & 3)
  xl = local_mul(l2, bl)
  return xg << 8 | xl


def symmetry_inv(g):
  def local_inv(g):
    return g ^ (g & 0x55) << 1
  def global_inv(g):
    return g ^ (~g >> 2 & (g & 1)) << 1
  li = local_inv(g & 255)
  gi = global_inv(g >> 8)
  # Commute local through global
  return gi << 8 | _commute_flat[gi << 8 | li]


def _rotate(g, x):
  n = int(np.sqrt(len(x)))
  x = jax.lax.cond(g & 1, lambda q: q.reshape(n,n)[:,::-1].T.reshape(-1), lambda q: q, x)
  return jax.lax.cond(g & 2, lambda q: q[::-1], lambda q: q, x)


@partial(jnp.vectorize, signature='(),(4,9)->(4,9)')
def transform_quads(g, quads):
  """Globally transform board quads by an element of D4"""
  assert g.dtype == np.int32
  assert quads.shape == (4,9)
  rotate = partial(_rotate, g)
  quads = jax.vmap(rotate, 0, 0)(quads)
  quads = jax.vmap(rotate, 1, 1)(quads)
  def reflect(q):
    return q.reshape(2,2,3,3)[::-1,::-1,::-1,::-1].swapaxes(0,1).swapaxes(2,3).reshape(4,9)
  quads = jax.lax.cond(g & 4, reflect, lambda q: q, quads)
  return quads


@partial(jnp.vectorize, signature='(),(4,9)->(4,9)')
def super_transform_quads(g, quads):
  """Transform board quads according to a supersymmetry"""
  assert g.dtype == np.int32
  quads = jax.vmap(_rotate)(g >> 2*jnp.arange(4) & 3, quads)
  return transform_quads(g >> 8, quads)


@partial(jnp.vectorize, signature='(),(2)->(2)')
def super_transform_board(g, board):
  """Transform a board according to a supersymmetry, keeping it in uint64 form"""
  return boards.quads_to_board(super_transform_quads(g, boards.board_to_quads(board)))


@partial(jnp.vectorize, signature='(),(4,w)->(4,w)')
def transform_state(g, x):
  """Globally transform an internal activation state by a rotation g (for now, we ignore reflections)"""
  return jnp.roll(x, g, axis=0)
