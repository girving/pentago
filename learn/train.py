#!/usr/bin/env python3
"""Pentago neural net training"""

import argparse
from dataclasses import dataclass
import datasets
from functools import partial
import equivariant as ev
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numbers
import optax
import os
import pickle
import platform
import timeit
import wandb


def pretty_info(info):
  def f(v):
    return v if isinstance(v, numbers.Integral) else f'{v:.3}'
  return ', '.join(f'{k} {f(v)}' for k, v in info.items())


def print_info(info):
  print(pretty_info(info))


def gpu_device():
  """The device we mostly want to run on."""
  return jax.devices('METAL' if platform.system() == 'Darwin' else 'gpu')[0]


@dataclass
class Config:
  # Architecture
  layers: int = 4
  width: int = 128
  mid: int = 128
  layer_scale: float = 1

  # Dataset
  slices: tuple[int] = (16,17,18)

  # Training
  batch: int = 1024 * 4
  valid_batch: int = 1024 * 16
  lr: float = 1e-3
  valid_every: int = 100
  save_every: int = 1000
  polyak: int = 100


def logits_fn(quads, *, config: Config):
  return ev.nf_net(quads, layers=config.layers, width=config.width, mid=config.mid, layer_scale=config.layer_scale)


def save(name, data, *, run):
  """Save some binary data to wandb."""
  path = os.path.join(run.dir, name)
  with open(path, 'wb') as f:
    pickle.dump(data, f)
  wandb.save(path, base_path=run.dir)


def exact_div(x, y):
  d = x // y
  assert x == d * y
  return d


def polyak_update(average, recent, *, polyak: int):
  """Update a Polyak average."""
  if not polyak:
    return recent
  b = jnp.asarray(1 / polyak)
  a = 1 - b
  return jax.tree.map(lambda x, y: a*x + b*y, average, recent)


def train(*,
          logits_fn,
          dataset,
          run,
          key=jax.random.PRNGKey(7)):
  config = run.config
  gpu = gpu_device()

  # Define network
  @hk.transform
  def loss_fn(data):
    quads = data['quads']
    values = data['value']
    batch = len(quads)
    assert quads.shape == (batch, 4, 9)
    assert values.shape == (batch,)
    labels = jax.nn.one_hot(values + 1, num_classes=3)
    logits, metrics = logits_fn(quads)
    loss = jnp.sum(labels * jax.nn.log_softmax(logits)) / -batch
    accuracy = (jnp.argmax(logits, axis=-1) == values + 1).astype(np.float32).mean()
    return loss, dict(loss=loss, accuracy=accuracy, **metrics)

  # Initialize
  key, key_ = jax.random.split(key)
  with jax.default_device(gpu):
    params = loss_fn.init(key_, next(dataset['train'].forever(batch=config.batch)))
  print(f'params = {sum(p.size for p in jax.tree.leaves(params)):,}')

  # Optimizer
  opt = optax.adamw(config.lr)
  opt_state = opt.init(params)
  polyak_params = params

  # Update step
  @partial(jax.jit, device=gpu)
  def update(params, opt_state, data):
    grads, metrics = jax.grad(lambda p: loss_fn.apply(p, None, data), has_aux=True)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics

  # Prepare for validation loss
  valid_data = next(dataset['valid'].forever(batch=config.valid_batch))
  @partial(jax.jit, device=gpu)
  def valid_metrics(params):
    _, metrics = loss_fn.apply(params, None, valid_data)
    return metrics

  # Estimate flops per board (see https://docs.jax.dev/en/latest/aot.html)
  if 0:  # Doesn't work with Metal, so probably need to build on CPU
    valid_metrics = valid_metrics.trace(params).lower().compile()
    flops = exact_div(valid_metrics.cost_analysis()['flops'], len(valid_metrics))
    print(f'flops/board = {flops}')

  # Train
  for step, data in enumerate(dataset['train'].forever(batch=config.batch)):
    e = dataset['train'].step_to_epoch(step, batch=config.batch)
    info = dict(epoch=e, samples=step*config.batch, time=timeit.default_timer())
    params, opt_state, ms = update(params, opt_state, data)
    polyak_params = polyak_update(polyak_params, params, polyak=config.polyak)
    for s,xs in ms.items():
      info['train/' + s] = np.mean(xs)
    if step % config.valid_every == 0:
      for s,xs in valid_metrics(params).items():
        info['valid/' + s] = np.mean(xs)
      for s,xs in valid_metrics(polyak_params).items():
        info['valid-polyak/' + s] = np.mean(xs)
    run.log(info, step=step)
    if step % config.save_every == 0:
      save(f'params-{step}.pkl', params, run=run)
      save(f'polyak-params-{step}.pkl', polyak_params, run=run)
      save(f'opt-state-{step}.pkl', opt_state, run=run)


def main():
  # Parse arguments
  parser = argparse.ArgumentParser(description='Pentago train')
  parser.add_argument('-f', '--fast', action='store_true')
  parser.add_argument('--lr', type=float, help='learning rate')
  options = parser.parse_args()

  # Configuration
  config = Config(save_every=2000)
  if options.fast:
    config.slices = 4,5,6
    config.layers = 1
    config.width = 16
    config.batch = config.valid_batch = 7
  if options.lr:
    config.lr = options.lr

  # Wandb setup
  run = wandb.init(
      entity='irving-personal',
      project='pentago',
      config=config)

  # Train
  dataset = datasets.sparse_dataset(counts=config.slices)
  train(logits_fn=partial(logits_fn, config=config), dataset=dataset, run=run)


if __name__ == '__main__':
  main()
