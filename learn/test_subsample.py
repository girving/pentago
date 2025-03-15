#!/usr/bin/env python3
"""Beam test"""

import jax
jax.config.update('jax_platform_name', 'cpu')

import asyncio
import boards
import datasets
import numpy as np
import pytest
import subsample
import tempfile
import trace


@pytest.mark.asyncio
async def test_subsample():
  max_slice = 5
  slices = range(max_slice + 1)
  prob = 0.0001
  if 0:
    super_path = index_path = 'gs://pentago-us-central1'
  else:
    super_path = '../data/edison/project/all'
    index_path = super_path + '-index'

  # Subsample with two different numbers of shards
  packs = {}
  with tempfile.TemporaryDirectory() as tmp:
    async def run(shards):
      path = f'{tmp}/{shards}'
      await subsample.subsample(slices=slices, index_path=index_path, super_path=super_path, prob=prob,
                                shards=shards, output_path=path)
      pieces = [np.load(f'{path}/subsample-shard{s}-of-{shards}.npz')['pack'] for s in range(shards)]
      packs[shards] = pack = np.sort(np.concatenate(pieces))
      assert pack.shape == (72,), pack.shape
      assert np.all(pack == np.unique(pack))
    await asyncio.gather(*[run(shards) for shards in (7, 10)])

  # Check data
  assert np.all(packs[7] == packs[10])
  board, value = boards.unpack_board_and_value(packs[7])
  board = np.array(board).view(np.uint64).squeeze(axis=-1)
  correct = {32:-1,7080255:0,146866518:-1,461839488:0,3736299:1,4326884075:0,4441833721:0,13123780691:-1,13171956210:-1,38663618566:0,38753861875:0,38947850379:-1,38759105241:1,90226229491:0,116139294783:0,141736084185:0,155526561954:1,347913585132:0,352201474048:1,696268161024:-1,1043677122973:0,1043679217965:0,1043820393297:0,1044107035281:0,1043682492422:1,1048067584861:0,1048258674699:0,1056565507125:0,2092510806025:0,3131031292826:0,3131034699294:0,3131036799254:0,3131063010828:0,3131174892597:1,6262348972284:1,9393104224526:0,9393128931571:0,9393135944004:0,9393237013316:-1,9393954037587:0,9394240291545:1,9741019840512:-1,10089166602243:0,28179329389065:0,28179423763337:0,28295345405954:0,28527507210294:0,28875208589313:0,29222961217554:0,31315466519986:0,56371504152657:0,59489608335361:0,65751755200929:0,282518669688840:-1,2533300560337340:0,7599863121445599:0,22799481707036675:1,22828696081661958:-1,68398423683433912:-1,68398497557840115:0,68398767243657783:-1,68399463019512296:-1,68426598621262661:0,68454777901940772:0,205195490381465739:0,615585782658301955:-1,615585813007569903:0,615588905097494825:0,1846757335088825784:0,1846757348063969526:0,1846758365876853159:0,1846776109102209441:0}
  datasets.correctness_test(zip(board, value), correct=correct)


def test_estimate():
  nonunique = {4: 804352, 5: 4841984}
  for max_slice, count in nonunique.items():
    approx = subsample.estimate(slices=range(max_slice+1))
    assert approx == count  # estimate is exact when compared against unique=False, prob=1


def test_safe_bernoulli():
  p = 1e-5
  n = 2**27

  @jax.jit
  def count(key):
    bits = subsample.safe_bernoulli(key, p, shape=(n,))
    return bits.astype(np.uint32).sum()

  c = count(jax.random.PRNGKey(7))
  std = np.sqrt(p * (1 - p) * n)
  assert np.abs(c - p * n) < std, f'count {c}, expected {p*n}, diff {c - p*n}, std {std}'


if __name__ == '__main__':
  asyncio.run(test_subsample())
  test_safe_bernoulli()
  test_estimate()
  trace.dump()
