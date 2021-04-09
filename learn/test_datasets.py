#!/usr/bin/env python3
"""Test datasets"""

import boards
import datasets
import jax
import numpy as np
import requests


_backend = 'https://us-central1-naml-148801.cloudfunctions.net/pentago'
_correct = {205251617442889971:1,348763127889:-1,286655715:0,77357582478:-1,347931279366:0,6610055528448:1,100860652:0,28179313459290:0,154629439737:0,1043879165952:-1,8208:1,2260:1,28179713949858:1,47531360257:0,1096076754944:-1,347940126776:1,347896152550:-1,348752314858:0,26542971:1,10436869626273:0,347940520422:1,695880255871:1,22818259603095555:0,348784165134:0,22799550714937425:0,47780880:0,615585800122663668:0,615613966232716722:-1,28179382534144:0,3246995406877:-1,1846766754806759586:0,115969426873:0,231933675659:1,28179286921494:1,2533279085559864:0}


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
