#!/usr/bin/env python3
"""Tests for batch_vmap"""

import jax
jax.config.update('jax_platform_name', 'cpu')

from batching import batch_vmap
import jax.numpy as jnp
import numpy as np


def test_batch_vmap():
  def f(key, info):
    o, s, p = info
    y = o + s * jnp.abs(jax.random.normal(key)) ** p
    return y + jax.random.uniform(key, (5,7))

  batch = 17
  sf = jnp.vectorize(f, signature='(2),(3)->(5,7)')
  bf = batch_vmap(batch)(f)
  key = jax.random.PRNGKey(7)
  k = jax.random.split(key, 100)
  i = jax.random.uniform(key, (100,3))
  y = sf(k, i)
  for n in 0, 5, batch, batch + 5, 100:
    assert np.allclose(y[:n], bf(k[:n], i[:n]))


def test_batch_vmap_tuple():
  def f(x):
    return x + 1, 2 * x

  batch = 17
  sf = jnp.vectorize(f, signature='()->(),()')
  bf = batch_vmap(batch)(f)
  x = jnp.arange(100)
  sy, sz = sf(x)
  for n in 0, 5, batch, batch + 5, 100:
    y, z = bf(x[:n])
    assert np.all(sy[:n] == y)
    assert np.all(sz[:n] == z)


if __name__ == '__main__':
  test_batch_vmap()
  test_batch_vmap_tuple()
