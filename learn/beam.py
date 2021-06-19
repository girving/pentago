#!/usr/bin/env python3
"""Beam job to extract random samples from .pentago data"""

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from batching import batch_vmap
import boards
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import os
from semicache import semicache
import supertensors as st
import symmetry


# TODO: Inherit GCP stuff from https://beam.apache.org/get-started/wordcount-example/#minimalwordcount-example


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
  board = symmetry.super_transform_board(jnp.arange(256), board) 
  keep = jax.vmap(lambda b: jax.random.bernoulli(b, p=prob))(board)
  black, white = ((data[:,None] >> jnp.arange(32)) & 1).astype(np.int8).reshape(2,256)
  value = (1-2*turn) * (black - white)
  return keep, board, value


def estimate(*, slices, prob=1.0):
  """Estimate the number of entries produced by subsample"""
  sections = st.descendent_sections(max(slices, default=0))
  return prob * 256 * sum(np.asarray(st.section_shape(sections[s]), dtype=np.uint64).prod(axis=-1).sum()
                          for s in slices)


def subsample(
    *,
    options=None,
    slices,
    index_path,
    super_path,
    prob,
    shards,
    output_path,
    unique=True,
):
  slices = tuple(slices)
  if not slices:
    return
  supers = semicache(lambda: st.Supertensors(
      max_slice=max(slices), index_path=index_path, super_path=super_path))
  expand = semicache(lambda: batch_vmap(16)(partial(expand_board, prob=prob)))

  def maybe_unique(xs):
    return (np.unique if unique else np.sort)(xs)

  def read_and_sample(section, I):
    board, data = supers().read_block(section, I)
    keep, board, value = map(np.asarray, expand()(board, data))
    return maybe_unique(boards.pack_board_and_value(board[keep], value[keep]))

  def pack_to_shard_(pack):
    return jax.random.randint(pack, shape=(), minval=0, maxval=shards)
  pack_to_shard = semicache(lambda: batch_vmap(64)(pack_to_shard_))

  def shatter(packs):
    """Split packs into shards"""
    s = np.asarray(pack_to_shard()(packs[:,None].view(np.uint32)))
    for i in range(shards):
      yield i, packs[s == i]

  def concat_and_shuffle(s, packs):
    """Assemble and shuffle a shard"""
    packs = maybe_unique(np.concatenate(packs + [np.zeros((0,), dtype=np.uint64)]))
    np.random.default_rng(seed=s).shuffle(packs)
    return s, packs

  def simple_write(s, pack):
    name = f'{output_path}/subsample-p{prob}-shard{s}-of-{shards}.npz'
    if not os.path.exists(name):  # Be idempotent
      np.savez(name, pack=pack)
    
  # Create directory if necessary
  os.makedirs(output_path, exist_ok=True)

  # Pipeline!
  with beam.Pipeline(options=options) as p:
    (p | beam.Create(np.concatenate([supers().sections[s] for s in slices]))
       | beam.FlatMap(lambda s: ((s,I) for I in st.section_all_blocks(s)))
       | beam.MapTuple(read_and_sample)
       | beam.FlatMap(shatter)
       | beam.GroupByKey()
       | beam.MapTuple(concat_and_shuffle)
       | beam.MapTuple(simple_write))


def main():
  options = PipelineOptions()
  
  # Parameters
  max_slice = 5
  base = '../data/edison/project/all'
  prob = 0.0001  # Probability of selecting each transformed board value
  shards = 10
  output_path = 'beam-out'

  # Subsample
  subsample(options=options, max_slice=max_slice, index_path=base+'-index', super_path=base, prob=prob,
            shards=shards, output_path=output_path)


if __name__ == '__main__':
  main()
