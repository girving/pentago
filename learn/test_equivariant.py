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


def test_embed():
  batch = 23
  width = 128
  np.random.seed(7)
  key = jax.random.PRNGKey(7)
  gs = np.random.randint(4, size=batch)
  boards = random_boards(size=batch, n=11)

  @hk.transform
  def net(boards):
    return ev.Embed(width)(boards)
  params = net.init(key, boards)
  if 0:
    # Construct fake parameters to make it easier to see patterns
    twos = (1 << jnp.arange(9)).astype(np.float32)
    twos = jnp.stack([jnp.zeros(9), twos, twos], axis=-1).reshape(9*3, 1)
    params = dict(embed=dict(embeddings=twos))
  h = jax.jit(lambda b: net.apply(params, None, b))
  hb = h(boards)
  gh = transform_state(gs, hb)
  hg = h(transform_board(gs, boards))
  if 0:
    print(f'g = {gs}')
    print(f'board =\n{show_board(boards[0])}')
    print(f'g board =\n{show_board(transform_board(gs[0], boards[0]))}')
    print(f'h =\n{h(boards)}')
    print(f'gh =\n{gh}')
    print(f'hg =\n{hg}')
    print(f'hb = {hb.mean()} ± {hb.std()}')
  assert np.all(gh == hg)
  assert np.allclose(hb.mean(), 0, atol=0.11)
  assert np.allclose(hb.std(), 1, atol=0.1)


def test_fourier():
  key = jax.random.PRNGKey(7)
  def norm(z):
    if isinstance(z, tuple):
      z0, z1, z2 = z
      return norm(z0) + 2*norm(z1) + norm(z2)
    return jnp.square(jnp.abs(z)).sum()
  for shape, axis in ((4,),0), ((7,4,11),1):
    s = jax.random.normal(key, shape)
    t = ev.fourier(s, axis=axis)
    u = ev.unfourier(t, axis=axis)
    if 0:
      print(f's = {s}, {norm(s)}')
      print(f'u = {u}, {norm(u)}')
      print(f't = {" ".join(jax.tree_map(str, t))}, {4*norm(t)}')
    assert np.allclose(t, np.moveaxis(np.fft.fft(s, axis=axis), axis, 0)[:3] / 2)
    assert np.allclose(s, u)
    assert np.allclose(norm(s), norm(t))


def test_convolve():
  def slow_convolve(x, y, *, axis, mul):
    x = jnp.moveaxis(x, axis, 0)
    y = jnp.moveaxis(y, axis, 0)
    return jnp.stack([sum(mul(x[j], y[(i-j)%4]) for j in range(4)) for i in range(4)], axis=axis)
  key0, key1 = jax.random.split(jax.random.PRNGKey(7))
  for sx, sy, mul in ((4,),(4,), jax.lax.mul), ((7,4,11),(11,4,13), jnp.matmul):
    axis = sx.index(4)
    x = jax.random.normal(key0, sx)
    y = jax.random.normal(key1, sy)
    z = ev.convolve(x, y, axis=axis, mul=mul)
    c = slow_convolve(x, y, axis=axis, mul=mul)
    assert np.allclose(z, c, rtol=1e-4)


def sws_test(net, *, shape):
  net = hk.transform(net)
  key = jax.random.PRNGKey(7)
  params = net.init(key, jnp.zeros(shape))
  x = jax.random.normal(key, shape) + 2
  y = net.apply(params, None, x)
  if 0:
    print(f'y = {np.mean(y)} ± {np.std(y)}')
  assert np.abs(np.mean(y)) < 0.02
  assert np.abs(np.std(y) - 1) < 0.02


def equivariant_net_test(net, *, batch=23, width=5):
  np.random.seed(7)
  key = jax.random.PRNGKey(7)
  gs = np.random.randint(4, size=batch)
  x = jax.random.normal(key, (batch, 4, width))
  net = hk.transform(net)
  params = net.init(key, x)
  h = lambda x: net.apply(params, None, x)
  hx = h(x)
  gh = transform_state(gs, hx)
  hg = h(transform_state(gs, x))
  if 0:
    print(f'g = {gs}')
    print(f'gh =\n{gh}')
    print(f'hg =\n{hg}')
  assert np.all(gh == hg)
  return x, hx


def test_sws_linear():
  sws_test(lambda x: ev.SwsLinear(230)(x), shape=(110, 170))


def test_sws_conv():
  sws_test(lambda x: ev.SwsConv(230)(x), shape=(110, 4, 170))
  equivariant_net_test(lambda x: ev.SwsConv(3)(x))


def test_equivariant_linear():
  equivariant_net_test(lambda x: ev.EquivariantLinear(3)(x))


def test_equivariant_block():
  equivariant_net_test(lambda x: ev.EquivariantBlock(mid=3)(x))


def test_nf_block():
  in_scale = 1.7
  y_scale = 3
  out_scale = np.sqrt(y_scale**2 - in_scale**2)
  def net(x):
    x = in_scale * x
    return ev.NFBlock(65)(x, in_scale=in_scale, out_scale=out_scale)
  x, y = equivariant_net_test(net, width=127)
  if 0:
    print(f'x = {x.mean()} ± {x.std()}')
    print(f'y = {y.mean()} ± {y.std()}')
  assert np.allclose(np.std(x), 1, rtol=0.01)
  assert np.allclose(np.std(y), y_scale, rtol=0.02)


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


def net_test(net, *, batch=23):
  np.random.seed(7)
  key = jax.random.PRNGKey(7)
  gs = np.random.randint(4, size=batch)
  boards = random_boards(size=batch, n=11)
  net = hk.transform(net)
  params = net.init(key, boards)
  h = jax.jit(lambda b: net.apply(params, None, b))
  hb, metrics = h(boards)
  hgb, _ = h(transform_board(gs, boards))
  if 0:
    print(f'g = {gs}')
    print(f'h =\n{hb}')
    print(f'hg =\n{hgb}')
  assert np.allclose(hb, hgb, atol=1e-5)
  return metrics


def test_invariant_net():
  net_test(lambda b: ev.invariant_net(b, layers=2, width=5, mid=3))


def test_nf_net():
  layers = 5
  layer_scale = 2
  def net(b):
    return ev.nf_net(b, layers=layers, width=64, mid=32, layer_scale=layer_scale)
  metrics = net_test(net, batch=128)
  means = metrics['means']
  stds = metrics['stds']
  assert means.shape == stds.shape == (layers+1,)
  expected = ev.nf_net_scale(jnp.arange(layers+1), layer_scale=layer_scale)
  if 0:
    print('layers:')
    for n in range(layers+1):
      print(f'  {n}: {means[n]:.3} ± {stds[n]:.3} vs. {expected[n]:.3}')
    print(f'ratios = {stds / expected}')
  assert np.allclose(means, 0, atol=0.6)
  assert np.allclose(stds, expected, rtol=0.08)
