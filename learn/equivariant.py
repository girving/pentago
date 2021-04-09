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


def equivariant_embed(boards, *, width):
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
  batch = boards.shape[0]
  assert boards.shape == (batch, 4, 9)
  q0, q1, q2, q3 = boards.swapaxes(0,1)
  q1 = q1.reshape(batch, 3, 3).swapaxes(1,2)[:,::-1,:].reshape(batch, 9)
  q2 = q2.reshape(batch, 3, 3)[:,::-1,:].swapaxes(1,2).reshape(batch, 9)
  q3 = q3[:, ::-1]
  rotated = jnp.stack([q0, q2, q3, q1], axis=1)  # Put quads in counterclockwise order
  count = (rotated.reshape(batch, 4*9) != 0).astype(np.int32).sum(axis=-1)
  init = hk.initializers.TruncatedNormal(stddev=0.01)
  x = hk.Embed(name='quads', vocab_size=9*3, embed_dim=width, w_init=init)(rotated + 3*jnp.arange(9)).sum(axis=-2)
  x += hk.Embed(name='count', vocab_size=19, embed_dim=width, w_init=init)(count)[:,None,:]
  assert x.shape == (batch, 4, width), x.shape
  return x


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


class EquivariantBlock:
  """An equivariant resnet block: x → x + linear(gelu(ev_linear(norm(x))))"""

  def __init__(self, *, mid):
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


class InvariantLogits:
  def __call__(self, x):
    """Invariant final logit layer

    Returns [batch,3]-shaped outcome logits on the value of the positions, with indices
    0,1,2 corresponding to outcomes -1,0,1 representing value for the player to move.
    """
    batch, _, width = x.shape
    assert x.shape == (batch, 4, width)
    x = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(x)
    x = jnp.mean(x, axis=1)
    logits = hk.Linear(3)(x)
    return logits


def invariant_net(boards, *, layers, width, mid):
  """A simple equivariant pentago reset"""
  x = equivariant_embed(boards, width=width)
  for layer in range(layers):
    x = EquivariantBlock(mid=mid)(x)
  logits = InvariantLogits()(x)
  return logits
