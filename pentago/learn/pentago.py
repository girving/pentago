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


_quad_threes = 3**tf.range(9)
_board_threes = tf.reshape(tf.bitwise.left_shift(
    3**tf.range(9, dtype=tf.int64), 16*tf.range(4, dtype=tf.int64)[:,None]), [-1])


def unpack_quads(quads):
  return tf.cast(quads[...,None], tf.int32) // _quad_threes % 3


def pack_quads(quads):
  return tf.reduce_sum(tf.cast(quads * _quad_threes, tf.uint16), axis=-1)


def unpack_boards(boards):
  assert boards.dtype == tf.int64, boards.dtype
  return tf.cast(boards[...,None] // _board_threes % 3, tf.int32)


def pack_boards(boards):
  assert boards.dtype == tf.int32
  return tf.reduce_sum(tf.cast(boards, tf.int64) * _board_threes, axis=-1)


def unpack_supers(supers):
  """Unpack a bunch of shape [2,32] supers into shape [256] classes.  Classes 0,1,2 are white-win, tie, black-win."""
  assert supers.dtype == tf.uint8
  assert supers.shape[-2:] == (2,32)
  supers = tf.cast(supers[...,None], tf.int32)
  bits = tf.bitwise.bitwise_and(tf.bitwise.right_shift(supers, tf.range(8)), 1)
  bits = tf.reshape(bits, tf.concat([tf.shape(bits)[:-2], [256]], axis=-1))
  black, white = tf.unstack(bits, axis=-2)
  return 1 + black - white


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
