#!/usr/bin/env python3
"""Test datasets"""

import boards
import datasets
import jax
import numpy as np


def test_sparse():
  steps = 7
  batch = 5
  dataset = datasets.sparse_dataset(counts=(4,5))
  count = 12544
  for split, percent in (('train', 99), ('valid', 1)):
    batches = dataset[split].batches(batch=batch)
    expected = count * percent // (100 * batch)
    assert np.allclose(batches, expected, rtol=0.15), f'{split}: batches {batches}, expected {expected}'
  dataset = dataset['train']

  # Test correctness
  correct = {45599423255150592:0,7599828667928872:-1,16698832846902:1,47287796230726188:-1,10621965:-1,281823156109313:1,227994731282497698:0,22809214386045090:-1,205195258027377133:1,17498115:1,1854359234067234822:0,30680781338050560:-1,13314:1,13217:0,19482401898496:1,1846757399508038475:0,2533283390947358:0,347987902469:1,70087270066225152:1,891816085:0,617274662581764098:-1,45598946260746321:1,22799821016532402:1,2599039329632310:-1,205476733285433533:0,1915718691540500534:0,319095522:0,70931694227030016:1,3942780108881:-1,6455767597056:0,3693796183803756544:-1,41400049729536:-1,4752:0,49545747:-1,7600172263543224:1}
  datasets.dataset_correctness_test(dataset, correct=correct, steps=steps, batch=batch)

  # Test that each epoch is different
  n = dataset.batches(batch=batch)
  data = [b for _, b in zip(range(3*n), dataset.forever(batch=batch))]
  assert np.any(data[0]['value'] != data[n]['value'])


if __name__ == '__main__':
  test_sparse()
