#!/usr/bin/env python3
"""Figure out what our probabilities should be"""

import jax
jax.config.update('jax_platform_name', 'cpu')

import numpy as np
import subsample
import supertensors as st

# Memorized plan
PROBS = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 0.3023919095768089, 10: 0.06713106773491546, 11: 0.017808530511928336, 12: 0.004642814578110831, 13: 0.0015155520744692097, 14: 0.000492378664769686, 15: 0.00018769277837386028, 16: 7.191682751273114e-05, 17: 3.420731672168119e-05, 18: 1.6517291722518227e-05}
SHARDS = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 4, 6: 26, 7: 138, 8: 794, 9: 1024, 10: 1024, 11: 1024, 12: 1024, 13: 1024, 14: 1024, 15: 1024, 16: 1024, 17: 1024, 18: 1024}


def plan():
  sections = st.descendent_sections(18)
  limit = 2**30
  shard_limit = 2**20

  print('estimates:')
  probs = {}
  all_shards = {}
  for slice in range(18+1):
    before = subsample.estimate(slices=[slice], sections=sections)
    prob = min(1.0, limit / before)
    after = int(prob * before)
    shards = (after - 1) // shard_limit + 1
    print(f'  {slice}: p{prob:.3} * {before:,} = {after:,} (shards {shards})')
    assert np.allclose(PROBS[slice], prob)
    probs[slice] = prob
    all_shards[slice] = shards

  if 1:
    print()
    print(f'PROBS = {probs}')
    print(f'SHARDS = {all_shards}')


if __name__ == '__main__':
  plan()
