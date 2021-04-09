#!/usr/bin/env python3
"""Test equivariances"""

from boards import Board
import equivariant as ev
from symmetry import transform_board, transform_state
from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def show_board(board):
  assert board.shape == (4,9)
  return str(board.reshape(2,2,3,3).swapaxes(1,2).reshape(6,6)[:,::-1].T)


def random_boards(*, size, n):
  boards = np.asarray([Board.random_board(n).quad_grid for _ in range(size)])
  assert boards.shape == (size, 4, 9)
  return boards


def test_equivariant_embed():
  batch = 23
  width = 5
  np.random.seed(7)
  key = jax.random.PRNGKey(7)
  gs = np.random.randint(4, size=batch)
  boards = random_boards(size=batch, n=11)

  @hk.transform
  def net(boards):
    return ev.equivariant_embed(boards, width=width)
  params = net.init(key, boards)
  if 0:
    # Construct fake parameters to make it easier to see patterns
    twos = (1 << jnp.arange(9)).astype(np.float32)
    twos = jnp.stack([jnp.zeros(9), twos, twos], axis=-1).reshape(9*3, 1)
    params = dict(embed=dict(embeddings=twos))
  h = lambda b: net.apply(params, None, b)
  gh = transform_state(gs, h(boards))
  hg = h(transform_board(gs, boards))
  if 0:
    print(f'g = {gs}')
    print(f'board =\n{show_board(boards[0])}')
    print(f'g board =\n{show_board(transform_board(gs[0], boards[0]))}')
    print(f'h =\n{h(boards)}')
    print(f'gh =\n{gh}')
    print(f'hg =\n{hg}')
  assert np.all(gh == hg)


def equivariant_net_test(net, *, batch=23, width=5):
  np.random.seed(7)
  key = jax.random.PRNGKey(7)
  gs = np.random.randint(4, size=batch)
  x = jax.random.normal(key, (batch, 4, width))
  net = hk.transform(net)
  params = net.init(key, x)
  h = lambda x: net.apply(params, None, x)
  gh = transform_state(gs, h(x))
  hg = h(transform_state(gs, x))
  if 1:
    print(f'g = {gs}')
    print(f'gh =\n{gh}')
    print(f'hg =\n{hg}')
  assert np.all(gh == hg)


def test_equivariant_linear():
  equivariant_net_test(lambda x: ev.EquivariantLinear(3)(x))


def test_equivariant_block():
  equivariant_net_test(lambda x: ev.EquivariantBlock(mid=3)(x))


def test_invariant_logits():
  batch = 23
  width = 5
  np.random.seed(7)
  key = jax.random.PRNGKey(7)
  gs = np.random.randint(4, size=batch)
  x = jax.random.normal(key, (batch, 4, width))

  @hk.transform
  def net(x):
    return ev.InvariantLogits()(x)
  params = net.init(key, x)
  h = lambda x: net.apply(params, None, x)
  assert np.allclose(h(x), h(transform_state(gs, x)))


def test_invariant_net():
  batch = 23
  np.random.seed(7)
  key = jax.random.PRNGKey(7)
  gs = np.random.randint(4, size=batch)
  boards = random_boards(size=batch, n=11)

  @hk.transform
  def net(b):
    return ev.invariant_net(b, layers=2, width=5, mid=3)
  params = net.init(key, boards)
  h = lambda b: net.apply(params, None, b)
  hb = h(boards)
  hgb = h(transform_board(gs, boards))
  if 1:
    print(f'g = {gs}')
    print(f'h =\n{hb}')
    print(f'hg =\n{hgb}')
  assert np.allclose(hb, hgb, atol=1e-5)
