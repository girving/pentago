"""Extract random samples from .pentago data"""

import aiofiles
import asyncio
from batching import batch_vmap
from bernoulli import safe_bernoulli
import boards
from functools import partial
from gcloud.aio.storage import Storage
import jax
import jax.numpy as jnp
import io
import numpy as np
import os
import re
import supertensors as st
import symmetry
import trace
from trace import scope, scoped


@jax.jit
def block_key(key, section, I):
  assert section.shape == (4,2)
  assert I.shape == (4,)
  key = jax.random.fold_in(key, section_sig(section))
  key = jax.random.fold_in(key, I << 8*jnp.arange(8))
  return key


def expand_board(board, data, *, prob):
  assert board.shape == (2,)
  assert data.shape == (16,)
  assert board.dtype == data.dtype == np.uint32
  turn = (boards.board_to_quads(board) != 0).astype(np.int32).sum() & 1
  board = symmetry.super_all_transform_board(board)
  keep = jax.vmap(lambda b: safe_bernoulli(b, p=prob))(board)
  black, white = ((data[:,None] >> jnp.arange(32)) & 1).astype(np.int8).reshape(2,256)
  value = (1-2*turn) * (black - white)
  return keep, board, value


def estimate(*, slices, prob=1, sections=None):
  """Estimate the number of entries produced by subsample"""
  if sections is None:
    sections = st.descendent_sections(max(slices, default=0))
  def reduce(shapes):
    return np.asarray(shapes).prod(axis=-1, dtype=int).sum(dtype=int)
  return prob * 256 * sum(reduce(st.section_shape(sections[s])) for s in slices)


@scoped
async def subsample(
    *,
    slices,
    index_path,
    super_path,
    prob,
    shards,
    output_path,
    unique=True,
):
  with scope('prep'):
    slices = tuple(slices)
    if not slices:
      return
    supers = st.Supertensors(max_slice=max(slices), index_path=index_path, super_path=super_path)
    expand = batch_vmap(64)(partial(expand_board, prob=prob))

    def maybe_unique(xs):
      return (np.unique if unique else np.sort)(xs)

    @batch_vmap(128)
    def pack_to_shard(pack):
      return jax.random.randint(pack, shape=(), minval=0, maxval=shards)

  @scoped
  async def read_sample_shatter(section, I, *, client, pieces):
    # Read
    board, data = await supers.read_block(section, I, client=client)

    # Subsample
    with scope('expand'):
      keep, board, value = map(np.asarray, expand(board, data))
    with scope('pack'):
      packs = maybe_unique(boards.pack_board_and_value(board[keep], value[keep]))

    # Split packs into shards
    s = np.asarray(pack_to_shard(packs[:,None].view(np.uint32)))
    for i in range(shards):
      pieces[i].append(packs[s == i])

  @scoped
  async def concat_shuffle_write(shard, packs, *, client):
    # Concatenate and shuffle a shard
    packs = maybe_unique(np.concatenate(packs + [np.zeros((0,), dtype=np.uint64)]))
    np.random.default_rng(seed=shard).shuffle(packs)

    # Prepare npz contents
    npz = io.BytesIO()
    np.savez(npz, pack=packs)
    npz = npz.getvalue()

    # Write output
    name = f'{output_path}/subsample-shard{shard}-of-{shards}.npz'
    if name.startswith('gs://'):
      m = re.match(r'^gs://([^/]+)/(.*)$', name)
      await client.upload(m.group(1), m.group(2), npz, timeout=1_000_000_000)
    else:
      os.makedirs(os.path.dirname(name), exist_ok=True)
      async with aiofiles.open(name, 'wb') as f:
        await f.write(npz)

  # Process everything
  async with Storage() as client:
    with scope('input'):
      pieces = [[] for _ in range(shards)]
      await asyncio.gather(*[
          read_sample_shatter(s, I, client=client, pieces=pieces)
          for slice in slices
          for s in supers.sections[slice]
          for I in st.section_all_blocks(s)])
    with scope('output'):
      await asyncio.gather(*[
          concat_shuffle_write(s, ps, client=client)
          for s, ps in enumerate(pieces)])
