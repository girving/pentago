#!/usr/bin/env python

from __future__ import division
from pentago import *
from other.core import *
from other.core.utility import Log
import subprocess
import tempfile
import shutil

def test_partition():
  random.seed(56565656)
  init_threads(-1,-1)
  slices = []
  for slice in xrange(7):
    # Grab half the sections of this slice
    sections = all_boards_sections(slice,8)
    random.shuffle(sections)
    slices.append(sections_t(slice,sections[:(len(sections)+1)//2]))
  for sections in descendent_sections([[4,4],[4,4],[4,5],[5,4]],35):
    if len(sections.sections):
      slices.append(sections)
  for sections in slices:
    for ranks in 1,2,3,5:
      for key in 0,1,17:
        with Log.scope('partition test: slice %d, ranks %d, key %d'%(sections.slice,ranks,key)):
          partition = random_partition_t(key,ranks,sections) if key else simple_partition_t(ranks,sections,False)
          partition_test(partition)

def test_simple_partition():
  init_threads(-1,-1)
  simple_partition_test()

def test_counts():
  init_threads(-1,-1)
  for slice in xrange(5):
    with Log.scope('counting slice %d'%slice):
      sections = sections_t(slice,all_boards_sections(slice,8))
      good_counts = meaningless_counts(all_boards(slice,1))
      good_nodes = sum(product(section_shape(s)) for s in sections.sections)
      for key in 0,1,17:
        with Log.scope('partition key %d'%key):
          partition = random_partition_t(key,1,sections) if key else simple_partition_t(1,sections,False)
          store = compacting_store_t(estimate_block_heap_size(partition,0))
          blocks = meaningless_block_store(partition,0,0,store)
          Log.write('blocks = %d, correct = %d'%(blocks.total_nodes,good_nodes))
          assert blocks.total_nodes==good_nodes
          bad_counts = sum_section_counts(sections.sections,blocks.section_counts)
          Log.write('bad counts  = %s\ngood counts = %s'%(bad_counts,good_counts))
          assert all(bad_counts==good_counts)

def test_fast_compress(local=False):
  init_threads(-1,-1)
  def compress_check(input):
    if local:
      compressed = local_fast_compress_test(input.copy())
      output = local_fast_uncompress_test(compressed) 
    else:
      buffer = empty(64+7*input.nbytes//6,dtype=uint8)
      compressed = buffer[:fast_compress(input.copy(),buffer,0)]
      output = empty_like(input)
      fast_uncompress(compressed,output,0)
    assert all(input==output)
    return compressed
  # Test a highly compressible sequence
  regular = arange(64//4*1873).view(uint64).reshape(-1,2,4)
  compressed = compress_check(regular)
  assert compressed[0]==1
  ratio = len(compressed)/regular.nbytes
  assert ratio < .314
  # Test various random (incompressible) sequences
  random.seed(18731)
  for n in 0,64,64*1873:
    bad = fromstring(random.bytes(n),dtype=uint64).reshape(-1,2,4)
    compressed = compress_check(bad)
    assert compressed[0]==0
    assert len(compressed)==bad.nbytes+1

def test_local_fast_compress():
  test_fast_compress(local=True)

def test_compacting_store():
  init_threads(-1,-1)
  compacting_store_test()
  
if __name__=='__main__':
  Log.configure('test',False,False,100)
  test_counts()
  test_partition()
  test_simple_partition()
  test_fast_compress()
  test_local_fast_compress()
