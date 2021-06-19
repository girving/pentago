#!/usr/bin/env python3
"""Beam test"""

import beam
import boards
import datasets
import numpy as np
import tempfile


def test_beam():
  max_slice = 5
  slices = range(max_slice + 1)
  prob = 0.0001
  super_path = '../data/edison/project/all'
  index_path = super_path + '-index'

  # Subsample with two different numbers of shards
  packs = {}
  with tempfile.TemporaryDirectory() as tmp:
    for shards in 7, 10:
      path = f'{tmp}/{shards}'
      beam.subsample(slices=slices, index_path=index_path, super_path=super_path, prob=prob,
                     shards=shards, output_path=path)
      pieces = [np.load(f'{path}/subsample-p{prob}-shard{s}-of-{shards}.npz')['pack'] for s in range(shards)]
      packs[shards] = pack = np.sort(np.concatenate(pieces))
      assert pack.shape == (79,), pack.shape
      assert np.all(pack == np.unique(pack))

  # Check data
  assert np.all(packs[7] == packs[10])
  board, value = boards.unpack_board_and_value(pack)
  board = np.array(board).view(np.uint64).squeeze(axis=-1)
  correct = {1242:0,728188:0,1195041:-1,1774089:-1,2164886:0,10682540:0,16122641:-1,19465662:-1,31922758:0,429985356:0,79389:1,4295428253:0,12885104329:0,12932813105:0,12973375488:-1,17181055555:-1,25782190324:0,25865945115:0,38681313286:0,38909509638:0,77645611011:-1,115966280427:0,116490049953:0,347894122155:-1,347904737469:-1,360876343323:0,579995763417:0,695786871228:0,696071358612:1,708680417361:1,734450221785:0,1159653556242:-1,1391601464130:0,2088261847188:0,3131041906940:0,3131748389142:0,3131031552057:1,6610416500763:-1,9393189093621:0,9393237001704:0,9393094080513:1,9741224706066:-1,11481309904896:0,15655187841267:-1,15655586961825:-1,18786331466172:-1,28179567096672:0,56359277494516:0,56359469193633:-1,844429241037640:0,844437820359768:0,863211217944579:-1,872720185307970:-1,900783586548138:0,2533283390947332:0,2533352099938316:0,2561454072594936:0,7599863029433781:-1,7600056357814299:-1,7606086433638566:-1,7629047329853718:-1,22808866210579893:0,22808866238891445:0,68398424495620258:1,68398535305986309:0,68398574055076674:0,68398805898362886:0,68454778335068403:-1,205195283792003316:0,205195336196751363:0,205195490242265097:0,205201520524984563:-1,615585786951237881:0,615586817748583512:0,1846757360853385288:0,1846757399508419532:0,1846757399511565017:0,1846757399651490209:0,1846757554137532555:0}
  datasets.correctness_test(zip(board, value), correct=correct)


def test_estimate():
  nonunique = {4: 804352, 5: 4841984}
  for max_slice, count in nonunique.items():
    approx = beam.estimate(slices=range(max_slice+1))
    assert approx == count  # estimate is exact when compared against unique=False, prob=1


if __name__ == '__main__':
  test_beam()
  test_estimate()
