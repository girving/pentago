#!/usr/bin/env python3
"""Drop high slices from shard files, byte for byte like strip_shard in shard.cc.

Stdlib only, so it runs on a stock cloud VM next to the bucket.  Each output is checked before it is
written: header fields other than max_slice unchanged, offsets consistent, and every kept group's
bytes identical to the input's.  Inputs already at max_slice are checked the same way and skipped.
The C++ tool (strip.cc) additionally decodes every group; tests check the two agree.

  usage: strip.py <max-slice> <output-dir> <shard-file> ...
"""

import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

MAGIC = b'pentago shard      \n'
HEADER = 36


def parse(data, path):
  assert data[:20] == MAGIC, path
  version, max_slice, shard_id, total = struct.unpack_from('<4I', data, 20)
  assert version == 2 and max_slice <= 18 and shard_id < total, path
  n = max_slice + 1
  offs = struct.unpack_from('<%dI' % (n + 1), data, HEADER)
  assert offs[0] == HEADER + 4 * (n + 1) and offs[n] == len(data), path
  assert all(offs[i] < offs[i + 1] for i in range(n)), path
  if path is not None:
    width = len(str(total - 1))
    expected = 'shard-%0*d-of-%0*d.pentago.shard' % (width, shard_id, width, total - 1)
    assert os.path.basename(path) == expected, path
  return max_slice, shard_id, total, offs


def strip(data, max_slice):
  old_max, _, _, offs = parse(data, None)
  assert 0 <= max_slice < old_max
  n = max_slice + 1
  shift = 4 * (old_max - max_slice)
  head = bytearray(data[:HEADER])
  struct.pack_into('<I', head, 24, max_slice)
  table = struct.pack('<%dI' % (n + 1), *[o - shift for o in offs[:n + 1]])
  return bytes(head) + table + data[offs[0]:offs[n]]


def process(args):
  path, max_slice, out_dir = args
  data = open(path, 'rb').read()
  old_max, shard_id, total, offs = parse(data, path)
  if old_max == max_slice:
    return 'skip'
  out = strip(data, max_slice)
  new_max, new_id, new_total, new_offs = parse(out, path)
  assert (new_max, new_id, new_total) == (max_slice, shard_id, total), path
  for s in range(max_slice + 1):
    assert out[new_offs[s]:new_offs[s + 1]] == data[offs[s]:offs[s + 1]], (path, s)
  with open(os.path.join(out_dir, os.path.basename(path)), 'wb') as f:
    f.write(out)
  return 'stripped'


def main():
  max_slice, out_dir, paths = int(sys.argv[1]), sys.argv[2], sys.argv[3:]
  with ProcessPoolExecutor() as pool:
    results = list(pool.map(process, [(p, max_slice, out_dir) for p in paths], chunksize=8))
  print('%d stripped, %d already at slice %d' % (results.count('stripped'), results.count('skip'),
                                                 max_slice))


if __name__ == '__main__':
  main()
