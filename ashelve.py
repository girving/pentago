#!/usr/bin/env python
'''An sqlite3-based shelf with locking features'''

from __future__ import division
import re
import cPickle as pickle
import sqlite3
import contextlib
from numpy import *

__all__ = ['ashelf','Locked']

# All keys are mapped to strings
keyers = {}
for t in int,long,int64,uint64,str:
  keyers[t] = repr

class Locked(RuntimeError):
  pass

# State bits
LOCKED = 1
SET = 2

class ashelf(object):
  def __init__(self,filename):
    # Open the database
    db = self.db = sqlite3.connect(filename)
    db.row_factory = sqlite3.Row
    # Create the table if it doesn't already exist
    try:
      db.execute('create table shelf (key text unique, state tinyint, value blob)')
    except sqlite3.OperationalError,e:
      if 'already exists' not in str(e):
        raise

  class Entry(object):
    def __init__(self,db,key,skey):
      self.db = db
      self.key = key
      self.skey = skey
      self.closed = False

    def __call__(self):
      if self.closed:
        raise ReferenceError()
      with self.db:
        for row in self.db.execute('select value from shelf where key = ? and state = ?',(self.skey,LOCKED|SET)):
          return pickle.loads(str(row['value']))
        raise KeyError(self.key)

    def set(self,value):
      if self.closed:
        raise ReferenceError()
      with self.db:
        self.db.execute('update shelf set state = ?, value = ? where key = ?',(LOCKED|SET,pickle.dumps(value),self.skey))

  @contextlib.contextmanager
  def lock(self,key):
    '''Lock an entry.  Raises Locked on failure.'''
    skey = keyers[type(key)](key)
    with self.db:
      for row in self.db.execute('select state from shelf where key = ?',(skey,)):
        if row['state']&LOCKED:
          raise Locked(key)
        self.db.execute('update shelf set state = ? where key = ?',(row['state']|LOCKED,skey))
        break
      else:
        self.db.execute('insert into shelf(key,state) values (?,?)',(skey,LOCKED))
    entry = self.Entry(self.db,key,skey)
    try:
      yield entry
    finally:
      entry.closed = True
      with self.db:
        for row in self.db.execute('select state from shelf where key = ?',(skey,)):
          if row['state']&SET:
            self.db.execute('update shelf set state = ? where key = ?',(SET,skey))
          else:
            self.db.execute('delete from shelf where key = ?',(skey,))

  def __getitem__(self,key):
    with self.lock(key) as entry:
      return entry()

  def __setitem__(self,key,value):
    with self.lock(key) as entry:
      entry.set(value)

  def dict(self):
    with self.db:
      d = {}
      for row in self.db.execute('select key,state,value from shelf'):
        key = eval(row['key'])
        if row['state']&LOCKED:
          raise Locked(key)
        d[key] = pickle.loads(str(row['value']))
      return d

  def keys(self):
    with self.db:
      s = set()
      for row in self.db.execute('select key,state from shelf'):
        key = eval(row['key'])
        if row['state']&LOCKED:
          raise Locked(key)
        s.add(key)
      return s

  def impl_keys(self):
    with self.db:
      return set(eval(row['key']) for row in self.db.execute('select key from shelf'))

  def dump(self,s=''):
    print '\ndump: %s'%s
    for line in self.db.iterdump():
      print '  '+line
