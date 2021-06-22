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
      assert pack.shape == (67,), pack.shape
      assert np.all(pack == np.unique(pack))
    await asyncio.gather(*[run(shards) for shards in (7, 10)])

  # Check data
  assert np.all(packs[7] == packs[10])
  board, value = boards.unpack_board_and_value(packs[7])
  board = np.array(board).view(np.uint64).squeeze(axis=-1)
  correct = {137735:-1,143343324:0,143921601:0,146866518:-1,477757937:0,15931108:1,859963888:1,4438426167:0,12885113823:0,17275421159:0,25775243273:1,26056654851:1,30073028608:0,38688325821:-1,43048960000:1,115965689868:0,347908870584:-1,579821961216:1,734476567257:0,1052267970569:0,1060888773094:0,1083191726359:0,1159816349106:0,2087370426534:-1,3131127306663:0,3131349664014:0,3131556692103:0,3143916453895:0,3144776155136:1,3363060252681:0,6266454608011:-1,9393270423714:-1,9393969365481:-1,9393189030033:1,9779641725762:-1,18786620479932:0,18786665103360:1,28179294591396:0,28179569246937:0,28179281021087:1,28192457293878:0,31310311981191:0,282170777731081:0,282518659073944:-1,287737899188251:-1,337833537962308:-1,844463584968749:-1,847555993145703:0,2533622683927259:0,7599851001022603:0,7599941195268115:0,7609217751318611:0,7628042307502098:-1,22799477695185141:0,22802642810044902:1,22827691049877558:0,205195266620850905:0,205195297536737379:0,205214044500984537:0,205223437318422600:-1,615586121990410519:0,615592036139335707:-1,615595167303009732:-1,615617084381331458:0,1846759409699586075:0,1846759409727897681:0,1846760453229851464:0}
  datasets.correctness_test(zip(board, value), correct=correct)


def test_estimate():
  nonunique = {4: 804352, 5: 4841984}
  for max_slice, count in nonunique.items():
    approx = subsample.estimate(slices=range(max_slice+1))
    assert approx == count  # estimate is exact when compared against unique=False, prob=1


def test_safe_bernoulli():
  p = 1e-6
  n = 2**26

  @jax.jit
  def count(key):
    bits = subsample.safe_bernoulli(key, p, shape=(n,))
    return bits.astype(np.uint32).sum()

  c = count(jax.random.PRNGKey(7))
  assert c == np.rint(p * n), f'count {c}, expected {p*n}'


if __name__ == '__main__':
  asyncio.run(test_subsample())
  test_safe_bernoulli()
  test_estimate()
  trace.dump()
