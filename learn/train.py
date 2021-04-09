#!/usr/bin/env python3
"""Pentago neural net training"""

import argparse
import datasets
import equivariant as ev
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import numbers
import optax


def pretty_info(info):
  def f(v):
    return v if isinstance(v, numbers.Integral) else f'{v:.3}'
  return ', '.join(f'{k} {f(v)}' for k, v in info.items())


def print_info(info):
  print(pretty_info(info))


def logits_fn(quads):
  layers = 4
  width = 128
  mid = 128
  return ev.invariant_net(quads, layers=layers, width=width, mid=mid)


def train(*,
          logits_fn,
          dataset,
          batch=64,
          lr=1e-3,
          slog=print_info,
          log_every=100,
          key=jax.random.PRNGKey(7)):
  # Define network
  @hk.transform
  def loss_and_metrics(data):
    quads = data['quads']
    values = data['value']
    assert quads.shape == (batch, 4, 9)
    assert values.shape == (batch,)
    labels = jax.nn.one_hot(values + 1, num_classes=3)
    logits = logits_fn(quads)
    loss = jnp.sum(labels * jax.nn.log_softmax(logits)) / -batch
    accuracy = (jnp.argmax(logits, axis=-1) == values + 1).astype(np.float32).mean()
    return loss, dict(loss=loss, accuracy=accuracy)

  # Initialize
  key, key_ = jax.random.split(key)
  params = loss_and_metrics.init(key_, next(dataset.forever(batch=batch)))
  print(f'params = {sum(p.size for p in jax.tree_leaves(params)):,}')

  # Optimizer
  opt = optax.adam(lr)
  opt_state = opt.init(params)

  # Update step
  @jax.jit
  def update(params, opt_state, data):
    grads, metrics = jax.grad(lambda p: loss_and_metrics.apply(p, None, data), has_aux=True)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, metrics

  # Train
  metrics = dict(loss=[], accuracy=[])
  for step, data in enumerate(dataset.forever(batch=batch)):
    params, opt_state, ms = update(params, opt_state, data)
    for s in metrics:
      metrics[s].append(ms[s])
    if step % log_every == 0:
      e = dataset.step_to_epoch(step, batch=batch)
      info = dict(step=step, epochs=e, samples=step*batch)
      for s, xs in metrics.items():
        info[s] = np.mean(xs)
        xs.clear()
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
  dataset = datasets.SparseData(seed=7, counts=(16,17,18))
  train(logits_fn=logits_fn, dataset=dataset, slog=slog)


if __name__ == '__main__':
  main()
