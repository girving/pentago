from __future__ import division,absolute_import,print_function,unicode_literals
from pentago import *
from geode import *
from cloud import *
import zlib

def test_cloud_slice():
  # Make a local copy of slice-2.pentago
  tmp = named_tmpfile(suffix='.pentago')
  correct = zlib.decompress(base64_decode(b'eJwrSM0rSUzPVyhOTS7JzM8rVlBQ4GJmYGAA4VAgLoDJlxakFpWk5hXnF4HlGZlABCMDGAgxMIIhB1gjIxQyMHhApBkioPR2NgiN11CYqcxQo0CGMiIZKgE1zAdKC7ATYyjUWEaosYQMjYUaCgJ/zasiohgYWJ5d2+LGfECwgVGRUQwk7h7o3vSAoZ6BM5ZB612Y6J2OW3YHQOKF0zLEPQLO/WJg1GwAmX0lfM2rjUfSf4PdwRIZhd1c8QYWqLkp+juPP2D8z8APMvda4J26H++zqrvaZggC5S5curVkX8z7agZG/QYWIP+3izY3itnEGRpyp+6j3NoJeo+PiQPlpH/qXZm5Un8bkYYKHYAZqv6c6f8DJnsGLqCh73/v/1eyXDYbJP7rnSrDTsNKWQZGrQMgAxl7RZnRQoAGRo6CUTAKRgZALj6YDih5gEqPic81zzxgcGeQigUqmPfmSoFwhpkAm9rUkBrp397OF83Wvt4SYgaU2rqz//2OfjUeBkYzjz9v516U3/a5FlzloBd1TAfEJEAGP8y9mP2AQZyBL5ZBgaFGKq6ccfoG/mK9BpCeyGU3aqXy+x8AyyWJ0IeNMWQYNo2xhEcK7KuJdZkzAtcs9sFq2CgYBaNghAIA8o/2dw=='))
  open(tmp.name,'wb').write(correct)

  # Compare local and cloud readers
  init_threads(-1,-1)
  for local,cloud in zip(open_supertensors(tmp.name),cloud_slice(2)):
    assert all(local.header.section==cloud.header.section)
    assert all(local.header.blocks==cloud.header.blocks)
    r0,r1,r2,r3 = map(xrange,local.header.blocks)
    for i0 in r0:
      for i1 in r1:
        for i2 in r2:
          for i3 in r3:
            b = i0,i1,i2,i3
            assert all(local.read_block(b)==cloud.read_block(b))

if __name__=='__main__':
  test_cloud_slice()
