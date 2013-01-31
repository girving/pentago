#!/usr/bin/env python

from __future__ import division
from pentago import *
from other.core import *
from other.core.utility import Log
import subprocess
import tempfile
import shutil

def run(cmd):
  Log.write(cmd)
  subprocess.check_call(cmd.split())

@cache
def mpirun():
  cmds = 'mpirun','aprun'
  devnull = open(os.devnull,'w')
  for cmd in 'mpirun','aprun':
    if not subprocess.call(['which',cmd],stdout=devnull,stderr=devnull):
      return cmd
  raise OSError('no mpirun found, tried %s'%', '.join(cmds))

class TmpDir(object):
  def __init__(self,suffix):
    self.name = tempfile.mkdtemp(suffix)

  def __del__(self):
    shutil.rmtree(self.name,ignore_errors=True)

def test_write(dir=None):
  if dir is None:
    tmpdir = TmpDir('test_write')
    dir = tmpdir.name
  init_threads(-1,-1)
  for slice in 3,4:
    # Write out meaningless data from MPI
    dir = '%s/write-%d'%(dir,slice)
    run('%s -n 2 endgame-mpi --threads 3 --dir %s --test write-%d'%(mpirun(),dir,slice))

    # Check
    with Log.scope('test write %d'%slice):
      # Do the same computation locally
      sections = all_boards_sections(slice,8)
      partition = simple_partition_t(1,sections_t(slice,sections),False)
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

def meaningless_test(key,dir=None):
  if dir is None:
    tmpdir = TmpDir('test_meaningless')
    dir = tmpdir.name
  for slice in 4,5:
    # Compute small count slices based on meaningless data
    dir = '%s/meaningless-s%d-r%d'%(dir,slice,key)
    run('%s -n 2 endgame-mpi --threads 3 --save 20 --memory 3G --meaningless %d --randomize %d 00000000 --dir %s'%(mpirun(),slice,key,dir))
    # Check validity
    run('%s/check --meaningless %d %s'%(os.path.dirname(__file__),slice,dir))

def test_meaningless_simple(dir=None):
  meaningless_test(0,dir)

def test_meaningless_random(dir=None):
  meaningless_test(17,dir)

if __name__=='__main__':
  Log.configure('test',False,False,100)
  if not os.path.exists('tmp'):
    os.mkdir('tmp')
  test_write('tmp')
  test_meaningless_simple('tmp')
  test_meaningless_random('tmp')
