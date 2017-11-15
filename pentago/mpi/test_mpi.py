#!/usr/bin/env python

from __future__ import division
from geode import *
from geode.utility import Log
import subprocess
import tempfile
import shutil
import sys

nop = False

def run(cmd):
  Log.write(cmd)
  if not nop:
    subprocess.check_call(cmd.split())

def check(dir,restart=0,reader_test=None,high_test=None):
  check = os.path.join(os.path.dirname(__file__),'../end/check')
  check = os.path.normpath(check)
  cmd = [check,'--restart=%d'%int(restart),dir]
  if reader_test is not None:
    cmd.extend(['--reader-test',str(reader_test)])
  if high_test is not None:
    cmd.extend(['--high-test',str(high_test)])
  run(' '.join(cmd))

@cache
def mpirun():
  if nop:
    return 'mpirun'
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
    check(wdir,restart=1)

def meaningless_test(key,dir=None,restart=False,extras=False):
  if dir is None:
    tmpdir = TmpDir('test_meaningless')
    dir = tmpdir.name
  for slice in 4,5:
    # Compute small count slices based on meaningless data
    wdir = '%s/meaningless-s%d-r%d'%(dir,slice,key)
    base = '%s -n 2 endgame-mpi --threads 3 --save 20 --memory 3G --meaningless %d --randomize %d 00000000'%(mpirun(),slice,key)
    run('%s --dir %s'%(base,wdir))
    check(wdir)
    if extras:
      check(wdir,reader_test=slice-1,high_test=slice-2)
    if restart:
      tdir = wdir+'-restart-test'
      run('%s --restart %s/slice-%d.pentago --dir %s --test restart'%(base,wdir,slice-1,tdir))
      rdir = wdir+'-restarted'
      run('%s --restart %s/slice-%d.pentago --dir %s'%(base,wdir,slice-1,rdir))
      check(rdir)

def test_meaningless_simple(dir=None):
  meaningless_test(0,dir)

def test_meaningless_random(dir=None):
  meaningless_test(17,dir,restart=1,extras=1)

if __name__=='__main__':
  Log.configure('test',False,False,100)
  if '-n' in sys.argv:
    nop = True
  elif not os.path.exists('tmp'):
    os.mkdir('tmp')
  test_write('tmp')
  test_meaningless_simple('tmp')
  test_meaningless_random('tmp')
