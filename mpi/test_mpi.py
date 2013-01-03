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
          blocks = meaningless_block_store(partition,0,0)
          Log.write('blocks = %d, correct = %d'%(blocks.total_nodes,good_nodes))
          assert blocks.total_nodes==good_nodes
          bad_counts = sum_section_counts(sections.sections,blocks.section_counts)
          Log.write('bad counts  = %s\ngood counts = %s'%(bad_counts,good_counts))
          assert all(bad_counts==good_counts)

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

def test_fast_compress():
  init_threads(-1,-1)
  def compress_check(input):
    buffer = empty(64+7*input.nbytes//6,dtype=uint8)
    compressed = buffer[:fast_compress(input.copy(),buffer,0)]
    output = empty_like(input)
    fast_uncompress(compressed,output,0)
    assert all(input==output)
    return compressed
  # Test a highly compressible sequence
  regular = arange(64//4*18731).view(uint64).reshape(-1,2,4)
  compressed = compress_check(regular)
  assert compressed[0]==1
  ratio = len(compressed)/regular.nbytes
  assert ratio < .31
  # Test various random (incompressible) sequences
  random.seed(18731)
  for n in 0,64,64*18731:
    bad = fromstring(random.bytes(n),dtype=uint64).reshape(-1,2,4)
    compressed = compress_check(bad)
    assert compressed[0]==0
    assert len(compressed)==bad.nbytes+1
  
if __name__=='__main__':
  Log.configure('test',False,False,100)
  if not os.path.exists('tmp'):
    os.mkdir('tmp')
  test_write('tmp')
  test_counts()
  test_meaningless_simple('tmp')
  test_meaningless_random('tmp')
  test_partition()
  test_simple_partition()
  test_fast_compress()
