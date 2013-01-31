#!/usr/bin/env python

from __future__ import division
from pentago import *
from other.core import *
from other.core.utility import Log
import subprocess
import tempfile
import shutil

nop = False

def run(cmd):
  Log.write(cmd)
  if not nop:
    subprocess.check_call(cmd.split())

def check(dir):
  check = os.path.join(os.path.dirname(__file__),'../end/check')
  check = os.path.normpath(check)
  run('%s %s'%(check,dir))

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
  for slice in 3,4:
    # Write out meaningless data from MPI
    wdir = '%s/write-%d'%(dir,slice)
    run('%s -n 2 endgame-mpi --threads 3 --dir %s --test write-%d'%(mpirun(),wdir,slice))
    check(wdir)

def meaningless_test(key,dir=None):
  if dir is None:
    tmpdir = TmpDir('test_meaningless')
    dir = tmpdir.name
  for slice in 4,5:
    # Compute small count slices based on meaningless data
    wdir = '%s/meaningless-s%d-r%d'%(dir,slice,key)
    run('%s -n 2 endgame-mpi --threads 3 --save 20 --memory 3G --meaningless %d --randomize %d 00000000 --dir %s'%(mpirun(),slice,key,wdir))
    check(wdir)

def test_meaningless_simple(dir=None):
  meaningless_test(0,dir)

def test_meaningless_random(dir=None):
  meaningless_test(17,dir)

if __name__=='__main__':
  Log.configure('test',False,False,100)
  if '-n' in sys.argv:
    nop = True
  elif not os.path.exists('tmp'):
    os.mkdir('tmp')
  test_write('tmp')
  test_meaningless_simple('tmp')
  test_meaningless_random('tmp')
