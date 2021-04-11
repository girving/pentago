"""Pentago data for neural net training"""

from functools import partial
from boards import board_to_quads
import jax
import jax.numpy as jnp
import numpy as np
import symmetry


@partial(jnp.vectorize, signature='(),(8,2)->()')
def super_extract(g, values):
  """Game value according to black"""
  assert values.dtype == np.uint32
  b, w = jax.vmap(lambda v: (v[g >> 5 & 7] >> (g & 31)) & 1)(values.reshape(2, 8)).astype(np.int32)
  return b - w


@jax.jit
def process_sparse_batch(step, data):
  """Extract data from a batch, picking random supers to extract along.

  Returned values are in terms of player to move."""
  batch, _, _ = data.shape
  assert data.dtype == np.uint32
  assert data.shape == (batch, 9, 2)
  key = jax.random.fold_in(jax.random.PRNGKey(80), step)
  g = jax.random.randint(key, (batch,), 0, 2048)
  data = jnp.asarray(data, dtype=data.dtype)
  board = symmetry.super_transform_board(g, data[:, 0])
  quads = board_to_quads(board)
  turn = jax.vmap(lambda q: (q != 0).astype(np.int32).sum() & 1)(quads)
  values = (1-2*turn) * super_extract(g, data[:,1:])
  return dict(board=board, quads=quads, value=values)


class SparseData:
  def __init__(self, *, seed, counts, base='../data/edison/project/all'):
    # Concatenate and shuffle all data
    data = []
    for count in counts:
      sparse = np.load(f'{base}/sparse-{count}.npy')
      assert sparse.dtype == np.uint64
      assert sparse.shape[1] == 9
      data.append(sparse)
    data = np.concatenate(data, axis=0)

    # Convert to 32-bit to avoid needing to enable jax 64-bit
    data = data.view(np.uint32)
    assert data.shape == (len(data), 18)
    data = data.reshape(len(data), 9, 2)
    self._data = jnp.asarray(data)

    @partial(jax.jit, static_argnums=1)
    def epoch(e, batch):
      # Shuffle differently per epoch
      key = jax.random.PRNGKey(17)
      key = jax.random.fold_in(key, e)
      data = jax.random.permutation(key, self._data)

      # Organize into batches
      data = data[:len(data) // batch * batch]
      return data.reshape(len(data) // batch, batch, 9, 2)
    self._epoch = lambda e, *, batch: epoch(e, batch)

  def batches(self, *, batch):
    return len(self._data) // batch

  def forever(self, *, batch):
    step = 0
    for e in range(100000):
      for raw in self._epoch(e, batch=batch):
        yield process_sparse_batch(step, raw)
        step += 1

  def step_to_epoch(self, step, *, batch):
    return step / self.batches(batch=batch)
