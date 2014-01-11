from __future__ import absolute_import

from . import pentago_core
from .pentago_core import *
from numpy import asarray

def large(n):
  s = str(n)
  return ''.join((',' if i and i%3==0 else '')+c for i,c in enumerate(reversed(s)))[::-1]

def report_thread_times(times,name=''):
  return pentago_core.report_thread_times(times,name)

def open_supertensors(path,io=IO):
  return open_supertensors_py(path,io)

def factorial(n):
  assert n>=0
  f = 1
  for i in xrange(2,n+1):
    f *= i
  return f

def binom(n,*k):
  k = asarray(k)
  sk = k.sum()
  if k.min()<0 or sk>n: return 0
  f = factorial(n)//factorial(n-sk)
  for i in k:
    f //= factorial(i)
  return f
