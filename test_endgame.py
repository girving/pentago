#!/usr/bin/env python

from __future__ import division
from other.core import *
from interface import *
import tempfile

def test_supertensor():
  Log.configure('test',0,1,0)
  init_thread_pools(-1,-1)

  # Choose tiny parameters
  section = (2,0),(0,2),(1,1),(1,1)
  block_size = 6
  filter = 0
  level = 6

  # Prepare for writing
  file = tempfile.NamedTemporaryFile(prefix='test',suffix='.pentago')
  writer = supertensor_writer_t(file.name,section,block_size,filter,level)
  blocks = writer.header.blocks
  assert all(blocks==(2,2,3,3))

  # Generate random data
  random.seed(873242)
  data = {}
  for i in xrange(blocks[0]):
    for j in xrange(blocks[1]):
      for k in xrange(blocks[2]):
        for l in xrange(blocks[3]):
          b = i,j,k,l
          shape = writer.header.block_shape(b)
          data[b] = fromstring(random.bytes(64*product(shape)),uint64).reshape(*hstack([shape,2,4]))

  # Write blocks out in arbitrary (hashed) order
  for b,block in data.iteritems():
    writer.write_block(b,block)
  writer.finalize()

  # Prepare for reading
  reader = supertensor_reader_t(file.name)
  assert all(reader.header.section==section)
  assert reader.header.block_size==block_size
  assert reader.header.filter==filter
  assert all(reader.header.blocks==blocks)

  # Read and verify data in a different order than we wrote it
  for i in xrange(blocks[0]):
    for j in xrange(blocks[1]):
      for k in xrange(blocks[2]):
        for l in xrange(blocks[3]):
          b = i,j,k,l
          block = reader.read_block(b)
          assert all(block==data[b])
  report_thread_times(True)

def test_popcounts_over_stabilizers():
  popcounts_over_stabilizers_test(1024)

if __name__=='__main__':
  test_supertensor()
  test_popcounts_over_stabilizers()
