#!/usr/bin/env python3
"""Test datasets"""

import boards
import datasets
import jax
import numpy as np
import requests


_backend = 'https://us-central1-naml-148801.cloudfunctions.net/pentago'
_correct = {410418695326336434:1,22799821005914319:-1,449515798:0,270678491406:-1,19146969514038:0,23362423066984531:1,430715795:0,2533626977714184:0,137650695370309632:0,19533513031680:-1,119345390125318144:1,2394789101854261248:1,206884107892949721:1,2091696979969:0,1096076754944:-1,4924686540469764096:1,835982768184754257:-1,5081161059532881:0,28875065131173:1,15762637350502646:0,270588252438:1,45603133820252994:1,22818259603095555:0,75998484235354112:0,22799550714937425:0,47780880:0,68400506698334220:0,3693514760361477068:-1,6691989028864:0,2462358751420614066:-1,615870380084692402:0,2393954070022914048:0,27866022696124422:1,3716314117514330841:1,4924714371825991689:0}


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
