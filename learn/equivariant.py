#!/usr/bin/env python3
'''Equivariant pentago networks

Let G = Z_4 be the group of global rotations.  (For now we are ignoring rotations,
as they're harder to deal with.)  Ignoring batch dimension, our interior activations
will have shape [4,width], corresponding to one width-dimension vector per quadrant.
If b is a board, h(b) is some internal layer of the network, and f(b) is the final
logits (shape [3]), we will enforce

1. g h(b) = h(gb)  # equivariance
2. f(b) = f(gb)    # invariance

where g ∈ G acts on internal activations of shape [4,width] by cyclic rotation.
'''

from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class Embed(hk.Module):
  """Embed [batch,4,9]-shaped boards equivariantly into [batch,4,width] activations

  We embed each quadrant by summing together embeddings at each position, using
  an embedding tensor of shape [9,3,width].  This tensor is rotated when applying
  it to the non-lower-left quadrants (or rather, those quadrants and rotated) so
  that equivariance holds after embedding.

  We also embed the total stone count, in particular to tell the network whose turn it is.

  Args:
    boards: A [batch,4,9]-shaped integer tensor with values in 0,1,2 (empty, black, white)
    width: Inner embedding dimension

  Returns:
    [batch,4,width]-shaped tensor
  """
  def __init__(self, width):
    super().__init__()
    self._width = width

  def __call__(self, boards):
    batch = boards.shape[0]
    width = self._width
    assert boards.shape == (batch, 4, 9)
    q0, q1, q2, q3 = boards.swapaxes(0,1)
    q1 = q1.reshape(batch, 3, 3).swapaxes(1,2)[:,::-1,:].reshape(batch, 9)
    q2 = q2.reshape(batch, 3, 3)[:,::-1,:].swapaxes(1,2).reshape(batch, 9)
    q3 = q3[:, ::-1]
    rotated = jnp.stack([q0, q2, q3, q1], axis=1)  # Put quads in counterclockwise order
    count = (rotated.reshape(batch, 4*9) != 0).astype(np.int32).sum(axis=-1)
    init = hk.initializers.RandomNormal(1 / np.sqrt(9+1))
    w_quads = hk.get_parameter('w_quads', shape=(9*3, width), init=init)
    w_count = hk.get_parameter('w_count', shape=(19, width), init=init)
    x = jax.nn.one_hot(rotated, 3).reshape(batch, 4, 9*3) @ w_quads
    x += (jax.nn.one_hot(count, 19) @ w_count)[:,None,:]
    assert x.shape == (batch, 4, width)
    return x


def fourier(s, *, axis, scale=0.5):
  """Size 4 Fourier transform, returning the complex middle result as a tuple"""
  assert s.dtype == np.float32
  s0, s1, s2, s3 = scale * jnp.moveaxis(s, axis, 0)
  s02 = s0 + s2
  s13 = s1 + s3
  t0 = s02 + s13
  t1 = s0 - s2, s3 - s1
  t2 = s02 - s13
  return t0, t1, t2


def unfourier(t, *, axis, scale=0.5):
  """Size 4 inverse Fourier transform"""
  t0, (t1r, t1i), t2 = t
  t0 = scale * t0
  t2 = scale * t2
  scale2 = 2 * scale
  t1r = scale2 * t1r
  t1i = scale2 * t1i
  s02 = t0 + t2
  s13 = t0 - t2
  s0 = s02 + t1r
  s2 = s02 - t1r
  s1 = s13 - t1i
  s3 = s13 + t1i
  return jnp.stack([s0, s1, s2, s3], axis=axis)


def convolve(x, y, *, axis, mul=jnp.matmul):
  """Convolve tensors of shape [a,4,b], [b,4,c] → shape [a,4,c]"""
  x0, (x1r, x1i), x2 = fourier(x, axis=axis, scale=1)
  y0, (y1r, y1i), y2 = fourier(y, axis=axis, scale=1)
  z0 = mul(x0, y0)
  z1r = mul(x1r, y1r) - mul(x1i, y1i)
  z1i = mul(x1r, y1i) + mul(x1i, y1r)
  z2 = mul(x2, y2)
  return unfourier((z0, (z1r, z1i), z2), axis=axis, scale=0.25)


def sws_standardize(w, *, gain, eps=1e-4, scale=1):
  """Scaled weight standardize a matrix, with optional scalar gain

  See Brock et al., https://arxiv.org/abs/2101.08692 for details.
  Code modified from https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py#L121.
  """
  n_in, n_out = w.shape
  w = w - jnp.mean(w, axis=0, keepdims=True)
  sum_sqr = jnp.square(w).sum(axis=0, keepdims=True)
  scale *= jax.lax.rsqrt(jnp.maximum(sum_sqr, eps))  # 1 / sqrt(n_in * var)
  if gain:
    scale *= hk.get_parameter('gain', shape=(n_out,), dtype=w.dtype, init=jnp.ones)
  return scale * w


class SwsLinear(hk.Module):
  """SWS linear layer"""
  def __init__(self, size, *, gain=True):
    super().__init__()
    self._size = size
    self._gain = gain

  def __call__(self, x):
    n_in = x.shape[-1]
    n_out = self._size
    w = hk.get_parameter('w', shape=(n_in, n_out), dtype=x.dtype,
                         init=hk.initializers.VarianceScaling(1, 'fan_in', 'normal'))
    bias = hk.get_parameter('bias', shape=(n_out,), dtype=x.dtype, init=jnp.zeros)
    w = sws_standardize(w, gain=self._gain)
    return x @ w + bias


class SwsConv(hk.Module):
  """SWS convolutional layer"""
  def __init__(self, size, *, gain=True, scale=1):
    super().__init__()
    self._size = size
    self._gain = gain
    self._scale = scale

  def __call__(self, x):
    batch, _, n_in = x.shape
    n_out = self._size
    assert x.shape == (batch, 4, n_in)
    w = hk.get_parameter('w', shape=(4*n_in, n_out), dtype=x.dtype,
                         init=hk.initializers.VarianceScaling(1, 'fan_in', 'normal'))
    bias = hk.get_parameter('bias', shape=(n_out,), dtype=x.dtype, init=jnp.zeros)
    w = sws_standardize(w, gain=self._gain, scale=self._scale)
    w = w.reshape(n_in, 4, n_out)
    y = convolve(x, w, axis=1) + bias
    assert y.shape == (batch, 4, n_out)
    return y


class EquivariantLinear:
  """An equivariant linear layer"""
  def __init__(self, size):
    self._size = size

  def __call__(self, x):
    batch, _, width = x.shape
    assert x.shape == (batch, 4, width)
    # Make a [batch,4,4*width]-shape tensor by rolling x 4 ways
    rolls = jnp.stack([jnp.roll(x, -k, axis=1) for k in range(4)], axis=1)
    rolls = rolls.reshape(batch, 4, 4*width)
    x = hk.Linear(self._size)(rolls)
    assert x.shape == (batch, 4, self._size)
    return x


class EquivariantBlock(hk.Module):
  """An equivariant resnet block: x → x + linear(gelu(ev_linear(norm(x))))"""

  def __init__(self, *, mid):
    super().__init__()
    self._mid = mid

  def __call__(self, x):
    mid = self._mid
    batch, _, width = x.shape
    assert x.shape == (batch, 4, width)
    y = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
    y = EquivariantLinear(self._mid)(y)
    y = jax.nn.gelu(y)
    y = hk.Linear(width)(y)
    return x + y


class NFBlock(hk.Module):
  """Normalizer-free resnet block

  See Brock et al., https://arxiv.org/abs/2101.08692 for details.
  """
  def __init__(self, mid):
    super().__init__()
    self._mid = mid

  def __call__(self, x, *, in_scale, out_scale):
    batch, _, width = x.shape
    assert x.shape == (batch, 4, width)
    gamma = np.sqrt(2 / (1 - 1/np.pi))
    y = SwsConv(self._mid, scale=gamma / in_scale)(x)
    y = jax.nn.relu(y)
    y = SwsConv(width, scale=out_scale)(y)
    return x + y


class InvariantLogits:
  def __call__(self, x, *, classes=3):
    """Invariant final logit layer

    Returns [batch,3]-shaped outcome logits on the value of the positions, with indices
    0,1,2 corresponding to outcomes -1,0,1 representing value for the player to move.
    """
    batch, _, width = x.shape
    assert x.shape == (batch, 4, width)
    x = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x)
    x = jnp.mean(x, axis=1)
    logits = hk.Linear(classes)(x)
    return logits


def invariant_net(boards, *, layers, width, mid):
  """A simple equivariant pentago reset"""
  x = Embed(width)(boards)
  for layer in range(layers):
    x = EquivariantBlock(mid=mid)(x)
  logits = InvariantLogits()(x)
  metrics = {}
  return logits, metrics


def nf_net_scale(layer, *, layer_scale):
  return jnp.sqrt(1 + layer*jnp.square(layer_scale))


def nf_net(boards, *, layers, width, mid, layer_scale, classes=3):
  """A normalizer-free equivariant pentago resnet"""
  scale = partial(nf_net_scale, layer_scale=layer_scale)
  means, stds = [], []
  def stat(x):
    means.append(jnp.mean(x))
    stds.append(jnp.std(x))
  x = Embed(width)(boards)
  stat(x)
  for layer in range(layers):
    x = NFBlock(mid)(x, in_scale=scale(layer), out_scale=layer_scale)
    stat(x)
  x = 0.25 / scale(layers) * x.sum(axis=1)
  logits = hk.Linear(classes)(x)
  metrics = dict(means=jnp.asarray(means),
                 stds=jnp.asarray(stds))
  return logits, metrics
