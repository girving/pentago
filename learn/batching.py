#!/usr/bin/env python3
"""Batched jitting"""

from functools import wraps
import jax
import jax.numpy as jnp
import timeit


def batch_vmap(batch):
  """jax.jit(jax.vmap(f)), but avoiding recompilation if shape[0] changes"""
  def inner(f):
    @jax.jit
    def batched(*args):
      return jax.vmap(f)(*args)

    @wraps(f)
    def g(*args):
      n = len(args[0])
      batches = max(1, -(-n // batch))
      padding = (0, batches * batch - n),
      last = n - (batches - 1) * batch
      def prep(x):
        assert len(x) == n
        mode = 'edge' if n else 'constant'
        x = jnp.pad(x, padding + ((0,0),)*(x.ndim-1), mode=mode)
        return x.reshape((batches,batch) + x.shape[1:])
      args = [prep(x) for x in args]
      ys = [batched(*(x[i] for x in args)) for i in range(batches)]
      def take(y):
        return jnp.concatenate(y[:-1] + (y[-1][:last],))
      if isinstance(ys[0], tuple):
        return tuple(take(y) for y in zip(*ys))
      else:
        return take(tuple(ys))

    return g
  return inner
