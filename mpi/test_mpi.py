#!/usr/bin/env python

from __future__ import division
from pentago import *
from other.core.utility import Log

def test_partition():
  partition_test()

def test_counts():
  for slice in xrange(5):
    with Log.scope('counting slice %d'%slice):
      sections = all_boards_sections(slice,8)
      partition = partition_t(1,8,slice,sections,False)
      blocks = meaningless_block_store(partition)
      good_blocks = sum(product(section_shape(s)) for s in sections)
      Log.write('blocks = %d, correct = %d'%(len(blocks.all_data),good_blocks))
      assert len(blocks.all_data)==good_blocks
      bad_counts = sum_section_counts(sections,blocks.section_counts)
      good_counts = meaningless_counts(all_boards(slice,1))
      Log.write('bad counts  = %s\ngood counts = %s'%(bad_counts,good_counts))
      assert all(bad_counts==good_counts)

if __name__=='__main__':
  Log.configure('test',False,False,100)
  test_counts()
  partition_test()
