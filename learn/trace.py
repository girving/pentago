"""Manual profiling for asyncio code"""

import asyncio
import contextlib
import contextvars
from dataclasses import dataclass
import functools
import numpy as np
import time


@dataclass
class _Stats:
  name: str
  times: list[float]
  kids: dict[str, '_Stats']


@dataclass
class _Frame:
  name: str
  stats: _Stats


_stats = _Stats('', [], {})
_root = _Frame('', _stats)
_stack = contextvars.ContextVar('stack', default=_root)


@contextlib.contextmanager
def scope(name):
  start = time.perf_counter() 
  parent = _stack.get()
  try:
    stats = parent.stats.kids[name]
  except KeyError:
    stats = parent.stats.kids[name] = _Stats(name, [], {})
  _stack.set(_Frame(name, stats))
  try:
    yield
  finally:
    stats.times.append(time.perf_counter() - start)
    _stack.set(parent)


def scoped(f):
  name = f.__name__
  if asyncio.iscoroutinefunction(f):
    @functools.wraps(f)
    async def wrapped(*args, **kwds):
      with scope(name):
        return await f(*args, **kwds)
  else:
    @functools.wraps(f)
    def wrapped(*args, **kwds):
      with scope(name):
        return f(*args, **kwds)
  return wrapped


def dump(*, verbose=()):
  assert _stack.get() is _root
  def show(s, indent):
    times = np.array(s.times)
    print(f'{indent}{s.name}: total {times.sum():.3} s, mean {times.mean():.3} s, count {len(times)}')
    if s.name in verbose:
      print(f'{indent}  times {times}')
    indent = indent + '  '
    for k in s.kids.values():
      show(k, indent)
  print('trace:')
  for s in _root.stats.kids.values():
    show(s, '  ')
