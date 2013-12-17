# Cloud utilities

from __future__ import division,print_function,unicode_literals,absolute_import
from pentago import *
from geode import *
import urllib2

def cloud_file(name,url):
  def pread(offset,size):
    req = urllib2.Request(url)
    req.headers['Range'] = 'bytes=%s-%s'%(offset,offset+size-1)
    data = urllib2.urlopen(req).read()
    assert type(data)==bytes and len(data)==size
    return asarray(buffer(data),dtype=uint8)
  return read_function(name,pread)

def cloud_slice(slice):
  base = 'http://582aa28f4f000f497ad5-81c103f827ca6373fd889208ea864720.r52.cf5.rackcdn.com/'
  name = 'slice-%d.pentago'%slice
  return open_supertensors(cloud_file(name,base+name))
