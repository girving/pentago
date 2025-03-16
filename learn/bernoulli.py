"""Random utility functions"""

import jax
import jax.numpy as jnp
import numpy as np


def safe_bernoulli(key, p, *, shape=None):
  """Bernoulli distribution that's accurate even for p ~ 1e-5"""
  assert key.shape == (2,)
  p = jnp.asarray(p)
  if shape is None:
    shape = p.shape
  hi = np.uint32(2**32 - 1)
  n = jax.random.randint(key, shape=shape, minval=0, maxval=hi, dtype=np.uint32)
  return n < jnp.clip(p, 0, 1) * hi
