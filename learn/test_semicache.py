#!/usr/bin/env python3
"""Tests for semicache"""

from semicache import semicache
import pickle


log = []
def f(x):
  log.append(x)
  return x + 7


def test_semicache():
  def check(f, x, hit):
    prev = log[:]
    y = f(x)
    assert y == x + 7
    assert log == prev + [x]*(1-hit)

  def checks(f):
    check(f, 3, False)
    check(f, 3, True)
    check(f, 9, False)
    check(f, 9, True)

  g = semicache(f)
  checks(g)
  h = pickle.loads(pickle.dumps(g))
  checks(h)
  check(g, 3, True)


if __name__ == '__main__':
  test_semicache()
