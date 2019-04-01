#!/usr/bin/env python3

from functools import lru_cache
import numpy as np
import os
import tensorflow as tf

tf.enable_eager_execution()

_ops = tf.load_op_library(
    os.path.normpath(os.path.join(__file__, '../../../bazel-bin/pentago/learn/libpentago_ops.so')))
count_boards = _ops.count_boards
block_counts = _ops.block_counts
block_info = _ops.block_info
pread = _ops.pread


def compact_str(v):
  def compact(a):
    if isinstance(a, dict):
      return '{%s}' % ','.join('%s:%s' % (compact(k), compact(v)) for k, v in a.items())
    if isinstance(a, tf.Tensor):
      return compact(a.numpy())
    if isinstance(a, (str, bytes)):
      return repr(a)
    try:
      return '[%s]' % ','.join(map(compact, a))
    except TypeError:
      return str(a)
  return compact(v)


def show_board(board, *, indent='', sep=' '):
  # Useful links:
  #   https://en.wikipedia.org/wiki/Box-drawing_character
  board = np.asarray(board)
  assert board.ndim == 2
  stones = np.asarray(['  ', '⚫', '⚪'])
  background = '\x1b[1;47m'
  clear = '\x1b[00m'
  return '\n'.join(indent + sep.join(background + ''.join(stones[c]) + clear for c in line.reshape(-1, 3))
                   for line in board.T[::-1])


_threes = 3**tf.range(9)


def unpack_quads(quads):
  return tf.cast(quads[...,None], tf.int32) // _threes[(None,)*quads.shape.rank] % 3


def pack_quads(quads):
  return tf.reduce_sum(tf.cast(quads * _threes, tf.uint16), axis=-1)


def rotate_packed_quad(quad, r):
  r = r % 4
  if r == 0:
    return quad
  quad = tf.reshape(unpack_quads(quad), [3,3])
  if r == 1:
    quad = tf.transpose(quad[:,::-1])
  elif r == 2:
    quad = quad[::-1,::-1]
  else:  # r == 3
    quad = tf.transpose(quad)[:,::-1]
  return pack_quads(tf.reshape(quad, [9]))


@lru_cache()
def base_path():
  base = os.path.expanduser('~/tmp/pentago-data')
  if not os.path.exists(base):
    base = 'gs://pentago/edison'
  return base


def load_indices(max_slice):
  return tf.stack([tf.io.read_file(base_path() + tf.strings.format('/slice-{}.pentago.index', (s,)))
                   for s in range(max_slice + 1)])


def read_block(index, slice, offset, size):
  path = base_path() + tf.strings.format('/slice-{}.pentago', slice)
  return _ops.unpack_block(index, pread(path, offset, size))
