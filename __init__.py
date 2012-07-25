from __future__ import absolute_import

from .interface import *

def large(n):
  s = str(n)
  return ''.join((',' if i and i%3==0 else '')+c for i,c in enumerate(reversed(s)))[::-1]
