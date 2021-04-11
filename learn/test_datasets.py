#!/usr/bin/env python3
"""Test datasets"""

import boards
import datasets
import jax
import numpy as np
import requests


_backend = 'https://us-central1-naml-148801.cloudfunctions.net/pentago'
_correct = {844433520198361:1,22799821005914319:-1,825003156742471680:0,562950096951719:-1,22799473113956730:0,23362423066984531:1,2057019129902333952:0,2533626977714184:0,45618776091138492:0,19533513031680:-1,119345390125318144:1,2394789101854261248:1,206884107892949721:1,5348024558096671:0,1096076754944:-1,287968967262289:1,146034200854:-1,56358566170930:0,28875065131173:1,103238474145:0,22799473117108647:1,45603133820252994:1,22818259603095555:0,75998484235354112:0,22799550714937425:0,47780880:0,1698243399651606:0,3693514760361477068:-1,205195258022081438:0,2462358751420614066:-1,45599100846080001:0,22821398921609243:0,27866022696124422:1,3716314117514330841:1,615592268486738649:0}


def test_sparse():
  steps = 7
  batch = 5
  cached = True
  if not cached:
    _correct.clear()
  dataset = datasets.SparseData(seed=7, counts=(4,5))
  assert dataset.batches(batch=batch) == 12544 // batch

  # Test correctness
  for step, b in zip(range(steps), dataset.forever(batch=batch)):
    for i in range(batch):
      board = np.asarray(b['board'][i]).view(np.uint64)[0]
      quads = b['quads'][i]
      value = b['value'][i]
      assert np.all(boards.Board.parse(str(board)).quad_grid == quads)
      if not cached:
        _correct[board] = requests.get(f'{_backend}/{board}').json()[str(board)]
      assert value == _correct[board], f'{board} → {value} != {_correct[board]}'
  assert len(_correct) == steps * batch
  if not cached:
    print(f'correct = {str(_correct).replace(" ","")}')

  # Test that each epoch is different
  n = dataset.batches(batch=batch)
  data = [b for _, b in zip(range(3*n), dataset.forever(batch=batch))]
  assert np.any(data[0]['value'] != data[n]['value'])
