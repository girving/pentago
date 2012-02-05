// A pentago player

/* Notes:
 *
 * 1. For rules, see http://en.wikipedia.org/wiki/Pentago
 *
 * 2. Pentago has an inconvenient number of spaces, namely 36 instead
 *    of 32.  We could potentially dodge this problem by templatizing
 *    over the value of the center space in each quadrant.  This is
 *    almost surely a win, but I'll stick to 2 64-bit integers for now
 *    and revisit that trick later.  It would be easy if I knew that
 *    the first four optimal moves were center moves, but I don't have
 *    a proof of that.
 */

#include <stdint.h>
#include <algorithm>
#include <iostream>
#include <Python.h>

using std::max;
using std::cout;
using std::endl;

namespace {

// A simple verion of static_assert
template<bool c> struct assertion_helper {};
template<> struct assertion_helper<true> { enum {value = 0}; };
#define static_assert(condition) \
  enum {assertion_helper_##__LINE__ = assertion_helper<(condition)!=0>::value};

// We require 64 bits for now to make interaction with python easier
static_assert(sizeof(long)==8);

// Each board is divided into 4 quadrants, and each quadrant is stored
// in one of the 16-bit quarters of a 64-bit int.  Within a quadrant,
// the state is packed in radix 3, which works since 3**9 < 2**16.
typedef uint64_t board_t;

// A side (i.e., the set of stones occupied by one player) is similarly
// broken into 4 quadrants, but each quadrant is packed in radix 2.
typedef uint64_t side_t;

// A single quadrant always fits into uint16_t, whether in radix 2 or 3.
typedef uint16_t quadrant_t;

template<int q> inline quadrant_t quadrant(uint64_t state) {
  static_assert(0<=q && q<4);
  return (state>>16*q)&0xffff;
}

inline uint64_t quadrants(quadrant_t q0, quadrant_t q1, quadrant_t q2, quadrant_t q3) {
  return q0|(uint64_t)q1<<16|(uint64_t)q2<<32|(uint64_t)q3<<48;
}

// Determine if one side has 5 in a row
inline bool won(side_t side) {
  /* To test whether a position is a win for a given player, we note that
   * there are 3*4*2+4+4 = 32 different ways of getting 5 in a row on the
   * board.  Thus, a 64-bit int can store a 2 bit field for each possible
   * method.  We then precompute a lookup table mapping each quadrant state
   * to the number of win-possibilities it contributes to.  28 of the ways
   * of winning occur between two boards, and 4 occur between 4, so a sum
   * and a few bit twiddling checks are sufficient to test whether 5 in a
   * row exists.  See helper for the precomputation code. */

  // static const uint64_t win_contributions[4][1<<9] = {...};
  #include "gen/win.h"

  uint64_t c = win_contributions[0][quadrant<0>(side)]
             + win_contributions[1][quadrant<1>(side)]
             + win_contributions[2][quadrant<2>(side)]
             + win_contributions[3][quadrant<3>(side)];
  return c&(c>>1)&0x55 // The first four ways of winning require contributions from three quadrants
      || c&(0xaaaaaaaaaaaaaaaa<<8); // The remaining 28 ways require contributions from only two
}

inline quadrant_t pack(quadrant_t side0, quadrant_t side1) {
  // static const uint16_t pack[1<<9] = {...};
  #include "gen/pack.h"
  return pack[side0]+2*pack[side1];
}

template<int s> inline quadrant_t unpack(quadrant_t state) {
  static_assert(0<=s && s<2);
  // static const uint16_t unpack[3**9][2] = {...};
  #include "gen/unpack.h"
  return unpack[state][s];
}

template<int s> inline side_t unpack(board_t board) {
  return quadrants(unpack<s>(quadrant<0>(board)),
                   unpack<s>(quadrant<1>(board)),
                   unpack<s>(quadrant<2>(board)),
                   unpack<s>(quadrant<3>(board)));
}

struct python_error {};

inline void check_interrupts() {
  if (PyErr_CheckSignals())
    throw python_error();
}

// Evaluate the current status of a board, returning one bit for whether each player has 5 in a row.
int status(board_t board) {
  const side_t side0 = unpack<0>(board),
               side1 = unpack<1>(board);
  const int won0 = won(side0),
            won1 = won(side1);
  return won0|won1<<1;
}

// Import precomputed move and rotation tables
// static const uint16_t rotations[512][2] = {...};
#include "gen/rotate.h"
// static const uint16_t move_offsets[(1<<9)+1] = {...};
// static const uint16_t move_flat[move_offsets[1<<9]] = {...}; 
#include "gen/move.h"

// We declare the move listing code as a huge macro in order to use it in multiple functions while taking advantage of gcc's variable size arrays.
// To use, declare side0 and side1 variables and call MOVES() to fill a board_t moves[total] array.
#define QR0(s,q,dir) rotations[quadrant<q>(side##s)][dir]
#define QR1(s,q) {QR0(s,q,0),QR0(s,q,1)}
#define QR2(s) {QR1(s,0),QR1(s,1),QR1(s,2),QR1(s,3)}
#define COUNT(q,qpp) \
  const int offset##q = move_offsets[quadrant<q>(filled)]; \
  const int count##q       = move_offsets[quadrant<q>(filled)+1]-offset##q; \
  const int total##qpp     = total##q + 8*count##q;
#define MOVE_QUAD(q,qr,dir,i) \
  quadrant_t side0_quad##i, side1_quad##i; \
  if (qr!=i) { \
    side0_quad##i = quadrant<i>(side1); \
    side1_quad##i = q==i?changed:quadrant<i>(side0); \
  } else { \
    side0_quad##i = rotated[1][i][dir]; \
    side1_quad##i = q==i?rotations[changed][dir]:rotated[0][i][dir]; \
  } \
  const quadrant_t both##i = pack(side0_quad##i,side1_quad##i);
#define MOVE(q,qr,dir) { \
  /* printf("sides: %lx %lx\n",(long)side0,(long)side1); */ \
  MOVE_QUAD(q,qr,dir,0) \
  MOVE_QUAD(q,qr,dir,1) \
  MOVE_QUAD(q,qr,dir,2) \
  MOVE_QUAD(q,qr,dir,3) \
  /* printf("boths: %x %x %x %x\n",both0,both1,both2,both3); */ \
  moves[total##q+8*i+2*qr+dir] = quadrants(both0,both1,both2,both3); \
}
#define COLLECT_MOVES(q) \
  for (int i=0;i<count##q;i++) { \
    const quadrant_t changed = quadrant<q>(side0)|move_flat[offset##q+i]; \
    MOVE(q,0,0) MOVE(q,0,1) \
    MOVE(q,1,0) MOVE(q,1,1) \
    MOVE(q,2,0) MOVE(q,2,1) \
    MOVE(q,3,0) MOVE(q,3,1) \
  }
#define MOVES() \
  /* Rotate all four quadrants left and right in preparation for move generation */ \
  const quadrant_t rotated[2][4][2] = {QR2(0),QR2(1)}; \
  /* Count the number of moves in each quadrant */ \
  const side_t filled = side0|side1; \
  const int total0 = 0; \
  COUNT(0,1) COUNT(1,2) COUNT(2,3) COUNT(3,4) \
  const int total = total4; \
  /* Collect the list of all possible moves.  Note that we repack with the sides */ \
  /* flipped so that it's still player 0's turn. */ \
  board_t moves[total]; \
  COLLECT_MOVES(0) COLLECT_MOVES(1) COLLECT_MOVES(2) COLLECT_MOVES(3)

// Evaluate a position down to a certain game tree depth.  If the depth is exceeded, assume a tie.
// We maintain the invariant that player 0 is always the player to move.  The result is 1 for a win
// 0 for tie, -1 for loss.
int evaluate(int depth, board_t board) {
  // Is the current position a win, loss, or immediate tie?
  const side_t side0 = unpack<0>(board),
               side1 = unpack<1>(board);
  const int won0 = won(side0),
            won1 = won(side1);
  if (won0 || won1 || !depth)
    return won0-won1;

  // Collect possible moves
  // const board_t moves[total] = {...};
  MOVES()
  if (!total)
    return 0;

  // No move ordering for now
  int best = -1;
  for (int i=0;i<total;i++) {
    int result = -evaluate(depth-1,moves[i]);
    if (result==1)
      return 1;
    best = max(best,result);
  }
  check_interrupts();
  return best;
}

// Python interface

void check_board(board_t board) {
  #define CHECK(q) \
    if (!(quadrant<q>(board)<(int)pow(3,9))) { \
      PyErr_Format(PyExc_ValueError,"quadrant %d has invalid value %d",q,quadrant<q>(board)); \
      throw python_error(); \
    }
  CHECK(0) CHECK(1) CHECK(2) CHECK(3)
}

PyObject* status_py(PyObject* self, PyObject* args, PyObject* kwargs) {
  board_t board;
  static const char* names[] = {"board",0};
  if (!PyArg_ParseTupleAndKeywords(args,kwargs,"l",(char**)names,&board,0))
    return 0;

  int result;
  try {
    check_board(board);
    result = status(board); 
  } catch (...) {
    return 0;
  }
  return PyInt_FromLong(result);
}

PyObject* evaluate_py(PyObject* self, PyObject* args, PyObject* kwargs) {
  int depth;
  board_t board;
  static const char* names[] = {"depth","board",0};
  if (!PyArg_ParseTupleAndKeywords(args,kwargs,"il",(char**)names,&depth,&board,0))
    return 0;

  if (depth<0) {
    PyErr_Format(PyExc_ValueError,"expected nonnegative depth, got %d",depth);
    return 0;
  }

  int result;
  try {
    check_board(board);
    result = evaluate(depth,board); 
  } catch (...) {
    return 0;
  }
  return PyInt_FromLong(result);
}

PyObject* moves_py(PyObject* self, PyObject* args, PyObject* kwargs) {
  board_t board;
  static const char* names[] = {"board",0};
  if (!PyArg_ParseTupleAndKeywords(args,kwargs,"l",(char**)names,&board,0))
    return 0;

  try {
    check_board(board);
    const side_t side0 = unpack<0>(board),
                 side1 = unpack<1>(board);
    // const board_t moves[total] = {...};
    MOVES()
    PyObject* tuple = PyTuple_New(total);
    for (int i=0;i<total;i++)
      PyTuple_SET_ITEM(tuple,i,PyInt_FromLong(moves[i]));
    return tuple;
  } catch (...) {
    return 0;
  }
}

PyMethodDef methods[] = {
  {"status",  (PyCFunction)status_py,METH_VARARGS|METH_KEYWORDS,"compute the current status of a board"},
  {"evaluate",(PyCFunction)evaluate_py,METH_VARARGS|METH_KEYWORDS,"evaluate(depth,board) evaluates the given packed Pentago position to the specified depth"},
  {"moves",   (PyCFunction)moves_py,METH_VARARGS|METH_KEYWORDS,"compute the set of player 0's possible moves starting from a given position"},
  {0} // sentinel
};

}

PyMODINIT_FUNC
initengine() {
  PyObject* m = Py_InitModule3("engine",methods,"A Pentago engine");
  if (!m) return;
}
