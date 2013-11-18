from __future__ import absolute_import

from . import pentago_core
from .pentago_core import *

def large(n):
  s = str(n)
  return ''.join((',' if i and i%3==0 else '')+c for i,c in enumerate(reversed(s)))[::-1]

def report_thread_times(times,name=''):
  return pentago_core.report_thread_times(times,name)

def open_supertensors(path,io=IO):
  return open_supertensors_py(path,io)
