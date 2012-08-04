#!/usr/bin/env python

from __future__ import division
import tempfile
import contextlib
from numpy import *
from ashelve import *

@contextlib.contextmanager
def expect_error(error):
  try:
    yield
    assert False
  except error:
    pass

def test_ashelf():
  # Create two aliases to the same shelf
  file = tempfile.NamedTemporaryFile(prefix='shelf',suffix='.db')
  shelf = ashelf(file.name)
  alias = ashelf(file.name)

  # The shelf should start off empty
  assert shelf.dict()=={}
  with expect_error(KeyError):
    shelf['a']

  # Test basic access
  shelf['a'] = 'b'
  alias['b'] = 'c'
  for s in shelf,alias:
    assert s['a']=='b'
    assert 'a' in s
    assert 'c' not in s
    assert s.dict()=={'a':'b','b':'c'}
    assert s.keys()==set('ab')
    assert s.impl_keys()==set('ab')

  # Test locking of existing entries
  for key in 'ac':
    with shelf.lock(key) as entry:
      for s in shelf,alias:
        assert s['b']=='c'
        with expect_error(Locked):
          s[key]
        with expect_error(Locked):
          with s.lock(key):
            assert False
      if key=='a':
        assert entry()=='b'
        assert bool(entry)
      else:
        assert not bool(entry)
        with expect_error(KeyError):
          entry()
      entry.set('d')
      assert entry()=='d'
    with expect_error(ReferenceError):
      entry()
  for s in shelf,alias:
    assert s.dict()=={'a':'d','b':'c','c':'d'}

  # Test that locks vanish if we throw
  for key in 'ae':
    with expect_error(StopIteration):
      with shelf.lock(key):
        for s in shelf,alias:
          assert s.impl_keys()==set('abc')|set(key)
        raise StopIteration
    for s in alias,shelf:
      if key=='a':
        assert s['a']=='d'
      else:
        with expect_error(KeyError):
          s['e']
      assert s.impl_keys()==set('abc')
  for s in shelf,alias:
    assert s.dict()=={'a':'d','b':'c','c':'d'}

  # Test different kind types
  for key in 0,long(1),int64(2),uint64(3):
    shelf[key] = 4

if __name__=='__main__':
  test_ashelf()
