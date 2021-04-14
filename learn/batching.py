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
      def prep(x):
        assert len(x) == n
        mode = 'edge' if n else 'constant'
        x = jnp.pad(x, padding + ((0,0),)*(x.ndim-1), mode=mode)
        return x.reshape((batches,batch) + x.shape[1:])
      args = [prep(x) for x in args]
      y = jnp.concatenate([batched(*(x[i] for x in args)) for i in range(batches)])
      return y[:n]

    return g
  return inner
