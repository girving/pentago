#!/usr/bin/env python

from __future__ import division
from geode import *
from pentago import *
import tempfile
import hashlib

def test_supertensor():
  Log.configure('test',0,1,0)
  init_threads(-1,-1)

  # Choose tiny parameters
  section = (2,0),(0,2),(1,1),(1,1)
  block_size = 6
  filter = 1 # Interleave
  level = 6

  # Prepare for writing
  file = tempfile.NamedTemporaryFile(prefix='test',suffix='.pentago')
  writer = supertensor_writer_t(file.name,section,block_size,filter,level)
  blocks = writer.header.blocks
  assert all(blocks==(2,2,3,3))

  # Generate random data
  key = 187131
  data = {}
  for i in xrange(blocks[0]):
    for j in xrange(blocks[1]):
      for k in xrange(blocks[2]):
        for l in xrange(blocks[3]):
          b = i,j,k,l
          shape = writer.header.block_shape(b)
          data[b] = random_supers(key,hstack([shape,2]).astype(int32))
          key += 1

  # Write blocks out in arbitrary (hashed) order
  for b,block in data.iteritems():
    writer.write_block(b,block.copy())
  writer.finalize()

  # Test exact hash to verify endian safety.  This relies on deterministic
  # compression, and therefore may fail in future.
  hash = hashlib.sha1(open(file.name).read()).hexdigest()
  assert hash=='618dc58cce95b3e4981a7817392b2441fbc41e50'

  # Prepare for reading
  reader0 = supertensor_reader_t(file.name)
  reader1, = open_supertensors(file.name)
  for reader in reader0,reader1:
    assert all(reader.header.section==section)
    assert reader.header.block_size==block_size
    assert reader.header.filter==filter
    assert all(reader.header.blocks==blocks)

  # Read and verify data in a different order than we wrote it
  for reader in reader0,reader1:
    for i in xrange(blocks[0]):
      for j in xrange(blocks[1]):
        for k in xrange(blocks[2]):
          for l in xrange(blocks[3]):
            b = i,j,k,l
            block = reader.read_block(b)
            assert all(block==data[b])
  report_thread_times(total_thread_times())

def test_popcounts_over_stabilizers():
  popcounts_over_stabilizers_test(1024)

if __name__=='__main__':
  test_supertensor()
  test_popcounts_over_stabilizers()
