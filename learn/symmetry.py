#!/usr/bin/env python3
"""Pentago symmetries"""

import boards
from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def _rotate(g, x):
  n = int(np.sqrt(len(x)))
  x = jax.lax.cond(g & 1, lambda q: q.reshape(n,n)[:,::-1].T.reshape(-1), lambda q: q, x)
  return jax.lax.cond(g & 2, lambda q: q[::-1], lambda q: q, x)


@partial(jnp.vectorize, signature='(),(4,9)->(4,9)')
def transform_board(g, board):
  """Globally transform one board by an element of D4"""
  assert g.dtype == np.int32
  assert board.shape == (4,9)
  rotate = partial(_rotate, g)
  board = jax.vmap(rotate, 0, 0)(board)
  board = jax.vmap(rotate, 1, 1)(board)
  board = jax.lax.cond(g & 4, lambda q: q.reshape(2,2,3,3)[:,::-1,:,::-1].reshape(4,9), lambda q: q, board)
  return board


@partial(jnp.vectorize, signature='(),(2)->(2)')
def super_transform_board(g, board):
  """Transform a board according to a supersymmetry, keeping it in uint64 form"""
  assert g.dtype == np.int32
  quads = boards.board_to_quads(board)
  quads = jax.vmap(_rotate)(g >> 2*jnp.arange(4) & 3, quads)
  quads = transform_board(g >> 8, quads)
  return boards.quads_to_board(quads)


@partial(jnp.vectorize, signature='(),(4,w)->(4,w)')
def transform_state(g, x):
  """Globally transform an internal activation state by a rotation g (for now, we ignore reflections)"""
  return jnp.roll(x, g, axis=0)
