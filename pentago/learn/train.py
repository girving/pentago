#!/usr/bin/env python3
"""Pentago training."""

import argparse
from dataclasses import dataclass
import model
import numpy as np
import os
import pentago
import tensorflow as tf


@dataclass
class Params(model.Params):
  max_slice: int = 7
  batch: int = 256
  shuffle_size: int = 256 * 8**4  # At least 256 blocks
  output: str = None


def parse_params():
  parser = argparse.ArgumentParser('Train a pentago model')
  for f in Params.__dataclass_fields__.values():
    parser.add_argument('--' + f.name.replace('_', '-'), type=f.type, default=f.default)
  args = parser.parse_args() 
  H = Params()
  for f in Params.__dataclass_fields__.values():
    setattr(H, f.name, getattr(args, f.name))
  return H


def data(H):
  indices = pentago.load_indices(H.max_slice)
  count = tf.cast(pentago.block_counts()[H.max_slice], tf.int64)
  index = tf.data.Dataset.range(count).shuffle(count).repeat()

  def expand(n):
    n = tf.cast(n, tf.int32)
    slice_, offset, size, *quads = pentago.block_info(indices, n)
    def piece(q):
      return tf.bitwise.left_shift(tf.cast(quads[q], tf.int64), 16*q)[(slice(None),) + (None,)*(3-q)]
    boards = tf.bitwise.bitwise_or(tf.bitwise.bitwise_or(piece(0), piece(1)),
                                   tf.bitwise.bitwise_or(piece(2), piece(3)))
    supers = pentago.read_block(n, slice_, offset, size)
    check = tf.reduce_all(tf.equal(tf.shape(boards), tf.shape(supers)[:4]))
    with tf.control_dependencies([tf.Assert(check, [tf.shape(boards), tf.shape(supers)])]):
      boards = tf.reshape(boards, [-1])
      supers = tf.reshape(supers, [-1, 2, 32])
    return tf.data.Dataset.from_tensor_slices((boards, supers))

  return index.flat_map(expand).shuffle(H.shuffle_size).batch(H.batch, drop_remainder=True)


def main():
  H = parse_params()
  assert H.output
  M = model.Model(H)
  optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
  summary_writer = tf.contrib.summary.create_file_writer(os.path.join(H.output, 'tb'), flush_millis=5000)

  @tf.function
  def train_step(step, boards, supers):
    boards = pentago.unpack_boards(boards)
    supers = pentago.unpack_supers(supers)
    with tf.GradientTape() as tape:
      logits = M(boards)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=supers))
    accuracy = tf.equal(tf.argmax(logits, axis=-1, output_type=tf.int32), supers)
    accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    grads = tape.gradient(loss, M.trainable_variables)
    optimizer.apply_gradients(zip(grads, M.trainable_variables))
    tf.print(tf.strings.format('step {}: loss {}, accuracy {}', (step, loss, accuracy)))
    with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
      tf.contrib.summary.scalar('loss', loss, step=step)
      tf.contrib.summary.scalar('accuracy', accuracy, step=step)

  for step, (boards, supers) in enumerate(data(H)):
    train_step(step, boards, supers)


if __name__ == '__main__':
  main()
