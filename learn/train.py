#!/usr/bin/env python3
"""Pentago neural net training"""

import argparse
import datasets
from functools import partial
import equivariant as ev
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numbers
import optax
import platform
import timeit


def pretty_info(info):
  def f(v):
    return v if isinstance(v, numbers.Integral) else f'{v:.3}'
  return ', '.join(f'{k} {f(v)}' for k, v in info.items())


def print_info(info):
  print(pretty_info(info))


def gpu_device():
  """The device we mostly want to run on."""
  return jax.devices('METAL' if platform.system() == 'Darwin' else 'gpu')[0]


def logits_fn(quads):
  layers = 4 * 1
  width = 128 * 1
  mid = 128 * 1
  layer_scale = 1
  return ev.nf_net(quads, layers=layers, width=width, mid=mid, layer_scale=layer_scale)


def train(*,
          logits_fn,
          dataset,
          batch=1024 * 4,
          valid_batch=1024 * 16,
          lr=1e-3,
          slog=print_info,
          log_every=100,
          key=jax.random.PRNGKey(7)):
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
    params = loss_fn.init(key_, next(dataset['train'].forever(batch=batch)))
  print(f'params = {sum(p.size for p in jax.tree.leaves(params)):,}')

  # Optimizer
  opt = optax.adamw(lr)
  opt_state = opt.init(params)

  # Update step
  @partial(jax.jit, device=gpu)
  def update(params, opt_state, data):
    grads, metrics = jax.grad(lambda p: loss_fn.apply(p, None, data), has_aux=True)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics

  # Prepare for validation loss
  valid_data = next(dataset['valid'].forever(batch=valid_batch))
  @partial(jax.jit, device=gpu)
  def valid_metrics(params):
    _, metrics = loss_fn.apply(params, None, valid_data)
    return metrics

  # Train
  metrics = dict(sps=[], loss=[], accuracy=[])
  for step, data in enumerate(dataset['train'].forever(batch=batch)):
    start = timeit.default_timer()
    params, opt_state, ms = update(params, opt_state, data)
    ms['sps'] = batch / (timeit.default_timer() - start)
    for s in metrics:
      metrics[s].append(ms[s])
    if step % log_every == 0:
      e = dataset['train'].step_to_epoch(step, batch=batch)
      info = dict(step=step, epochs=e, samples=step*batch)
      for s, xs in metrics.items():
        info[s] = np.mean(xs)
        xs.clear()
      vms = valid_metrics(params)
      for s in 'loss', 'accuracy':
        info['valid_' + s] = np.mean(vms[s])
      slog(info)


def main():
  # Parse arguments
  parser = argparse.ArgumentParser(description='Pentago train')
  parser.add_argument('--log', type=str, help='log file')
  options = parser.parse_args()

  # Logging
  if options.log:
    log_file = open(options.log, 'w')
    def slog(info):
      s = pretty_info(info)
      print(s)
      print(s, file=log_file)
      log_file.flush()
  else:
    slog = print_info

  # Train
  dataset = datasets.sparse_dataset(counts=(16,17,18))
  train(logits_fn=logits_fn, dataset=dataset, slog=slog)


if __name__ == '__main__':
  main()
