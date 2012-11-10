from __future__ import absolute_import

from .interface import *
import libpentago

def large(n):
  s = str(n)
  return ''.join((',' if i and i%3==0 else '')+c for i,c in enumerate(reversed(s)))[::-1]

def report_thread_times(times,name=''):
  return libpentago.report_thread_times(times,name)
