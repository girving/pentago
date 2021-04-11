# Symmetry tests

import boards
import jax
import numpy as np
import symmetry as sym


def test_group():
  """Verify that symmetry_mul and symmetry_inv define a group"""
  mul = jax.jit(sym.symmetry_mul)

  # Identity and inverses
  g = np.arange(2048)
  i = sym.symmetry_inv(g)
  assert np.all(mul(0, g) == g)
  assert np.all(mul(g, 0) == g)
  assert np.all(mul(g, i) == 0)
  assert np.all(mul(i, g) == 0)

  # Random associativity test
  rng = np.random.default_rng(seed=7)
  a, b, c = rng.integers(2048, size=(3,1000))
  assert np.all(mul(a, mul(b, c)) == mul(mul(a, b), c))


def test_action():
  """Test action of the group on boards"""
  mul = jax.jit(sym.symmetry_mul)
  act = jax.jit(sym.super_transform_quads)
  rng = np.random.default_rng(seed=7)
  for shape in (), (1000,), (3, 7): 
    print(f'shape {shape}')
    g, h = rng.integers(2048, size=(2,) + shape)
    b = rng.integers(3, size=shape + (4,9), dtype=np.int32)
    if 0:
      print(f'g {g}, h {h}, gh {mul(g, h)}')
      print(f'b =\n{boards.show_quads(b)}')
      print(f'h b =\n{boards.show_quads(act(h, b))}')
    assert np.all(act(mul(g, h), b) == act(g, act(h, b)))
