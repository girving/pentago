#!/usr/bin/env python

from __future__ import division
from pentago import *
from other.core import *
from other.core.utility import Log
import subprocess

def test_partition():
  init_threads(-1,-1)
  partition_test()

def test_counts():
  init_threads(-1,-1)
  for slice in xrange(5):
    with Log.scope('counting slice %d'%slice):
      sections = all_boards_sections(slice,8)
      partition = partition_t(1,slice,sections,False)
      blocks = meaningless_block_store(partition,0,0)
      good_blocks = sum(product(section_shape(s)) for s in sections)
      Log.write('blocks = %d, correct = %d'%(blocks.nodes(),good_blocks))
      assert blocks.nodes()==good_blocks
      bad_counts = sum_section_counts(sections,blocks.section_counts)
      good_counts = meaningless_counts(all_boards(slice,1))
      Log.write('bad counts  = %s\ngood counts = %s'%(bad_counts,good_counts))
      assert all(bad_counts==good_counts)

def run(cmd):
  Log.write(cmd)
  subprocess.check_call(cmd.split())

@cache
def mpirun():
  cmds = 'mpirun','aprun'
  for cmd in 'mpirun','aprun':
    if not subprocess.call(['which','-s',cmd]):
      return cmd
  raise OSError('no mpirun found, tried %s'%cmds)

def test_write(tmpdir):
  init_threads(-1,-1)
  for slice in 3,4:
    # Write out meaningless data from MPI
    dir = '%s/write-%d'%(tmpdir,slice)
    run('%s -n 2 endgame-mpi --threads 3 --dir %s --test write-%d'%(mpirun(),dir,slice))

    # Check
    with Log.scope('test write %d'%slice):
      # Do the same computation locally
      sections = all_boards_sections(slice,8)
      partition = partition_t(1,slice,sections,False)
      blocks = meaningless_block_store(partition,0,0)

      # Compare counts
      counts = load('%s/counts-%d.npy'%(dir,slice))
      assert all(counts[:,0].copy().view(uint8).reshape(-1,4,2)==sections)
      correct = blocks.section_counts.copy()
      correct[:,1] = correct[:,2]-correct[:,1].copy()
      if slice&1:
        correct[:,:2] = correct[:,:2][:,::-1].copy()
      assert all(counts[:,1:]==correct)

      # Compare sparse
      sparse = load('%s/sparse-%d.npy'%(dir,slice))
      boards = sparse[:,0].copy()
      data = sparse[:,1:].copy().reshape(-1,2,4)
      assert len(data)==256*len(sections)
      compare_blocks_with_sparse_samples(blocks,boards,data)

      # Compare full data
      slice_file = '%s/slice-%d.pentago'%(dir,slice)
      readers = open_supertensors(slice_file)
      assert len(readers)==len(sections)
      total = 20+3*4
      for reader,section in zip(readers,sections):
        assert all(reader.header.section==section)
        total += reader.total_size()
      assert total==os.stat(slice_file).st_size
      compare_blocks_with_supertensors(blocks,readers)

def test_meaningless(tmpdir):
  for slice in 4,5:
    # Compute small count slices based on meaningless data
    dir = '%s/meaningless-%d'%(tmpdir,slice)
    run('%s -n 2 endgame-mpi --threads 3 --save 20 --memory 3G --meaningless %d 00000000 --dir %s'%(mpirun(),slice,dir))
    # Check validity
    run('%s/check --meaningless %d %s'%(os.path.dirname(__file__),slice,dir))

if __name__=='__main__':
  Log.configure('test',False,False,100)
  if not os.path.exists('tmp'):
    os.mkdir('tmp')
  test_write('tmp')
  test_counts()
  test_meaningless('tmp')
  partition_test()
