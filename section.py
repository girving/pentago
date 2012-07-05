#!/usr/bin/env python
'''Filename definitions and utilities for endgame traversal'''

from __future__ import division
from other.core import *
from other.core.value import parser
from pentago import *

data_dir = PropManager.add('dir','data').set_help('top level data directory')
symmetries = PropManager.add('symmetries',8).set_help('number of symmetries used to standardize sections')

def show_section(s):
  return '%d-'%sum(s)+''.join(map(str,ravel(s)))

def section_file(s):
  return os.path.join(data_dir(),'section-%s.pentago'%show_section(s))

def sparse_file(s):
  return os.path.join(data_dir(),'sparse-%s.try'%show_section(s))

def child_section(section,q):
  if sum(section[q])==9:
    return None
  child = section.copy()
  turn = int(sum(section))&1
  child[q,turn] += 1
  return child

def parent_section(section,q):
  parent = section.copy()
  turn = not (int(sum(section))&1)
  if parent[q,turn]:
    parent[q,turn] -= 1
    return parent
  return None

def child_section_shape(section,q):
  '''The shape of a child section of the given section, or () if the child doesn't exist'''
  child = child_section(section,q)
  if child is None:
    return zeros(4,int)
  return section_shape(child)

def child_reader(readers,section,q):
  child = child_section(section,q)
  if child is None:
    return None
  child = standardize_section(child,symmetries())
  reader, = [r for r in readers if all(r.header.section==child)]
  return reader

def section_children(section):
  turn = int(sum(section))&1
  children = []
  for q in xrange(4):
    child = child_section(section,q)
    if child is not None:
      child = standardize_section(child,symmetries())
      if not [() for c in children if all(child==c)]:
        children.append(child)
  return children

def section_parents(section):
  turn = int(sum(section))&1
  parents = []
  for q in xrange(4):
    parent = parent_section(section,q)
    if parent is not None:
      parent = standardize_section(parent,symmetries())
      if not [() for c in parents if all(parent==c)]:
        parents.append(parent)
  return parents

exist_set = set()
def exists(file):
  return file in exist_set or os.path.exists(file)
