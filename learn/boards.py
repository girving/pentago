"""Pentago board operations"""

from functools import partial
import jax.numpy as jnp
import numpy as np
import re

_win_rays = np.asarray([[0,6],[1,6],[2,6],[3,6],[4,6],[5,6],[6,6],[7,6],[8,6],[9,6],[10,6],[11,6],[0,1],[1,1],[6,1],[7,1],[12,1],[13,1],[18,1],[19,1],[24,1],[25,1],[30,1],[31,1],[4,5],[5,5],[10,5],[11,5],[0,7],[1,7],[6,7],[7,7]])
_board_re = re.compile(r'^(\d+)(m?)$')


@partial(jnp.vectorize, signature='(2)->(4,9)')
def board_to_quads(board):
  """We use [2]-shaped uint32 input to avoid needing to enable jax 64-bit"""
  assert board.dtype == np.uint32, board
  s = 16*jnp.arange(2)
  t = 3**jnp.arange(9)
  return (board[:,None] >> s & 0xffff).astype(np.int32).reshape(4,1) // t % 3


@partial(jnp.vectorize, signature='(4,9)->(2)')
def quads_to_board(quads):
  assert quads.dtype == np.int32
  s = 16*jnp.arange(2)
  t = 3**jnp.arange(9)
  return ((quads * t).sum(axis=-1).reshape(2,2) << s).astype(np.uint32).sum(axis=-1)


def show_quads(quads):
  assert quads.shape == (4,9)
  return str(quads.reshape(2,2,3,3).swapaxes(1,2).reshape(6,6)[:,::-1].T)


def random_quads(*, size, n):
  quads = np.asarray([Board.random_board(n).quad_grid for _ in range(size)])
  assert quads.shape == (size, 4, 9)
  return quads


class Board:
  """Mirror of high_board_t"""

  def __init__(self, quads, middle):
    quads = np.asarray(quads)
    if quads.shape != (4,) or middle not in (0,1):
      raise ValueError(f'Invalid board: quads {quads}, middle {middle}')
    self.quads = quads
    self.middle = middle

  def __str__(self):
    board = (self.quads << 16*np.arange(4)).sum()
    return f'{board}{"m"*self.middle}'

  @staticmethod
  def parse(s):
    m = _board_re.match(s)
    if not m:
      raise ValueError(f"Invalid board '{s}'")
    quads = int(m.group(1)) >> 16*np.arange(4) & 0xffff
    assert np.all(quads < 3**9)
    return Board(quads, len(m.group(2)))

  @staticmethod
  def random_board(n):
    """Random board with n stones and !middle"""
    assert 0 <= n <= 36
    count1 = n // 2
    count0 = n - count1
    i = np.arange(36)
    grid = (i < count1).astype(int) + (i < count0 + count1)
    np.random.shuffle(grid)
    board = Board((grid.reshape(4, 9) * 3**np.arange(9)).sum(axis=-1), middle=False)
    assert board.counts == (count0, count1)
    return board

  @property
  def quad_grid(self):
    """[4,9] int tensor of 0,1,2 values (empty, black, white)"""
    return self.quads[..., None] // 3**np.arange(9) % 3

  @property
  def grid(self):
    """[6,6] tensor of 0,1,2 values (empty, black, white)"""
    q = self.quad_grid
    return q.reshape(2,2,3,3).swapaxes(1,2).reshape(6,6)

  @property
  def count(self):
    return (self.quad_grid != 0).sum()

  @property
  def counts(self):
    q = self.quad_grid
    return tuple((q == k).sum() for k in (1,2))

  @property
  def turn(self):
    return (self.count + self.middle) & 1

  def place(self, x, y):
    """Place a stone at the given location"""
    grid = self.grid
    if self.middle or grid[x,y]:
      raise ValueError(f'Bad place: {self}, xy {x} {y}')
    quads = self.quads.copy()
    quads[x//3*2+y//3] += (1+self.turn) * 3**(3*(x%3)+y%3)
    return Board(quads, middle=True)

  def rotate(self, qx, qy, d):
    """Rotate the given quadrant left (d=1) or right (d=-1)"""
    if not self.middle or qx not in (0,1) or qy not in (0,1) or abs(d) != 1:
      raise ValueError(f'Bad rotate: {name}, q {qx} {qy}, d {d}')
    q = 2*qx + qy
    quad = self.quad_grid[q].reshape(3,3)
    quad = quad[:,::-1].T if d==1 else quad.T[:,::-1]
    quads = self.quads.copy()
    quads[q] = (quad.reshape(9) * 3**np.arange(9)).sum()
    return Board(quads, middle=False)

  def won(self, side):
    """Does the given side have 5 in a row?"""
    v = side + 1
    grid = self.grid.reshape(36)
    s, d = _win_rays.T[..., None]
    rays = s + d*np.arange(5)
    return (grid[rays] == v).all(axis=-1).any(axis=0)

  @property
  def done(self):
    """Is the game over?"""
    return self.won(0) or self.won(1) or (self.count==36 and not self.middle)

  @property
  def immediate_value(self):
    """Assuming the game is over, what is the current player's results?  1=win, 0=tie, -1=loss"""
    turn = self.turn
    win = self.won(turn)
    lose = self.won(1-turn)
    if win or lose:
      if win and lose:
        return 0
      return 1 if win else -1
    if self.count == 36:
      return 0
    raise ValueError(f'Board {name}: immediate_value called when board isn\'t done')

  def moves(self):
    if self.middle:
      return [self.rotate(qx,qy,d) for qx in (0,1) for qy in (0,1) for d in (-1,1)]
    else:
      grid = self.grid
      return [self.place(x,y) for x in range(6) for y in range(6) if not grid[x,y]]
