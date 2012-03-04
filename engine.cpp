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

#include <other/core/array/Array.h>
#include <other/core/array/NdArray.h>
#include <other/core/python/module.h>
#include <other/core/utility/format.h>
#include <other/core/utility/interrupts.h>

using std::max;
using std::cout;
using std::endl;
using std::swap;
using namespace other;

namespace {

/***************** Statistics ****************/

#define STAT(...) __VA_ARGS__

uint64_t expanded_nodes;
uint64_t total_lookups;
uint64_t successful_lookups;

void clear_stats() {
  expanded_nodes = total_lookups = successful_lookups = 0;
}

/***************** Engine ****************/

// Each board is divided into 4 quadrants, and each quadrant is stored
// in one of the 16-bit quarters of a 64-bit int.  Within a quadrant,
// the state is packed in radix 3, which works since 3**9 < 2**16.
typedef uint64_t board_t;

// A side (i.e., the set of stones occupied by one player) is similarly
// broken into 4 quadrants, but each quadrant is packed in radix 2.
typedef uint64_t side_t;

// A single quadrant always fits into uint16_t, whether in radix 2 or 3.
typedef uint16_t quadrant_t;

// The two low bits of the score are the result: loss 0, tie 1, win 2.
// Above these are 8 bits giving the depth of the result, e.g., >=36 for known and 0 for completely unknown.
typedef uint16_t score_t;

inline score_t score(int depth, int value) {
  return depth<<2|value;
}

inline score_t exact_score(int value) {
  return score(36,value);
}

inline quadrant_t quadrant(uint64_t state, int q) {
  assert(0<=q && q<4);
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

  uint64_t c = win_contributions[0][quadrant(side,0)]
             + win_contributions[1][quadrant(side,1)]
             + win_contributions[2][quadrant(side,2)]
             + win_contributions[3][quadrant(side,3)];
  return c&(c>>1)&0x55 // The first four ways of winning require contributions from three quadrants
      || c&(0xaaaaaaaaaaaaaaaa<<8); // The remaining 28 ways require contributions from only two
}

inline quadrant_t pack(quadrant_t side0, quadrant_t side1) {
  // static const uint16_t pack[1<<9] = {...};
  #include "gen/pack.h"
  return pack[side0]+2*pack[side1];
}

inline quadrant_t unpack(quadrant_t state, int s) {
  assert(0<=s && s<2);
  // static const uint16_t unpack[3**9][2] = {...};
  #include "gen/unpack.h"
  return unpack[state][s];
}

inline side_t unpack(board_t board, int s) {
  return quadrants(unpack(quadrant(board,0),s),
                   unpack(quadrant(board,1),s),
                   unpack(quadrant(board,2),s),
                   unpack(quadrant(board,3),s));
}

inline uint64_t hash(board_t key) {
  // Invertible hash function from http://www.concentric.net/~ttwang/tech/inthash.htm
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

/* Hashed transposition table.
 *
 * Notes:
 * 1. Collisions are resolved simply: the entry with greater depth wins.
 * 2. Since our hash is bijective, we store only the high order bits of the hash to detect collisions.
 */
const int score_bits = 10;
const int score_mask = (1<<score_bits)-1;
int table_bits = 0;
uint64_t table_mask = 0;
uint64_t* table = 0;

void init_table(int bits) {
  if (bits<1 || bits>30)
    throw ValueError(format("expected 1<=bits<=30, got bits = %d",bits));
  if (64-bits+10>64)
    throw ValueError(format("bits = %d is too small, the high order hash bits won't fit",bits));
  free(table);
  table_bits = bits;
  table_mask = (1<<table_bits)-1;
  table = (uint64_t*)calloc(1L<<bits,sizeof(uint64_t));
}

score_t lookup(board_t board) {
  STAT(total_lookups++);
  uint64_t h = hash(board);
  uint64_t entry = table[h&table_mask];
  if (entry>>score_bits==h>>table_bits) {
    STAT(successful_lookups++);
    return entry&score_mask;
  }
  return 1;
}

void store(board_t board, score_t score) {
  uint64_t h = hash(board);
  uint64_t& entry = table[h&table_mask];
  if (entry>>score_bits==h>>table_bits || (entry&score_mask)>>2 <= score>>2)
    entry = h>>table_bits<<score_bits|score;
}

void check_board(board_t board) {
  #define CHECK(q) \
    if (!(quadrant(board,q)<(int)pow(3.,9.))) \
      throw ValueError(format("quadrant %d has invalid value %d",q,quadrant(board,q)));
  CHECK(0) CHECK(1) CHECK(2) CHECK(3)
}

// Evaluate the current status of a board, returning one bit for whether each player has 5 in a row.
inline int status(board_t board) {
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  if ((side0|side1)==0x1ff01ff01ff01ff)
    return 3; // The board is full, so immediate tie
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
// To use, invoke MOVES(board) to fill a board_t moves[total] array.
#define QR0(s,q,dir) rotations[quadrant(side##s,q)][dir]
#define QR1(s,q) {QR0(s,q,0),QR0(s,q,1)}
#define QR2(s) {QR1(s,0),QR1(s,1),QR1(s,2),QR1(s,3)}
#define COUNT(q,qpp) \
  const int offset##q = move_offsets[quadrant(filled,q)]; \
  const int count##q       = move_offsets[quadrant(filled,q)+1]-offset##q; \
  const int total##qpp     = total##q + 8*count##q;
#define MOVE_QUAD(q,qr,dir,i) \
  quadrant_t side0_quad##i, side1_quad##i; \
  if (qr!=i) { \
    side0_quad##i = quadrant(side1,i); \
    side1_quad##i = q==i?changed:quadrant(side0,i); \
  } else { \
    side0_quad##i = rotated[1][i][dir]; \
    side1_quad##i = q==i?rotations[changed][dir]:rotated[0][i][dir]; \
  } \
  const quadrant_t both##i = pack(side0_quad##i,side1_quad##i);
#define MOVE(q,qr,dir) { \
  MOVE_QUAD(q,qr,dir,0) \
  MOVE_QUAD(q,qr,dir,1) \
  MOVE_QUAD(q,qr,dir,2) \
  MOVE_QUAD(q,qr,dir,3) \
  moves[total##q+8*i+2*qr+dir] = quadrants(both0,both1,both2,both3); \
}
#define COLLECT_MOVES(q) \
  for (int i=0;i<count##q;i++) { \
    const quadrant_t changed = quadrant(side0,q)|move_flat[offset##q+i]; \
    MOVE(q,0,0) MOVE(q,0,1) \
    MOVE(q,1,0) MOVE(q,1,1) \
    MOVE(q,2,0) MOVE(q,2,1) \
    MOVE(q,3,0) MOVE(q,3,1) \
  }
#define MOVES(board) \
  /* Unpack sides */ \
  const side_t side0 = unpack(board,0), \
               side1 = unpack(board,1); \
  /* Rotate all four quadrants left and right in preparation for move generation */ \
  const quadrant_t rotated[2][4][2] = {QR2(0),QR2(1)}; \
  /* Count the number of moves in each quadrant */ \
  const side_t filled = side0|side1; \
  const int total0 = 0; \
  COUNT(0,1) COUNT(1,2) COUNT(2,3) COUNT(3,4) \
  int total = total4; /* Leave mutable to allow in-place pruning of the list */ \
  /* Collect the list of all possible moves.  Note that we repack with the sides */ \
  /* flipped so that it's still player 0's turn. */ \
  board_t moves[total]; \
  COLLECT_MOVES(0) COLLECT_MOVES(1) COLLECT_MOVES(2) COLLECT_MOVES(3)

// Evaluate position based only on current state and transposition table
inline score_t quick_evaluate(board_t board) {
  // Is the current position a win, loss, or immediate tie?
  const int st = status(board);
  if (st)
    return exact_score(1+(st&1)-(st>>1));
  // Did we already compute this value?
  return lookup(board);
}

// Flip a score and increment its depth by one
inline score_t lift(score_t sc) {
  return score(1+(sc>>2),2-(sc&3));
}

// Same as evaluate, but assume board status and transposition table have supplied no information.
score_t evaluate_recurse(int depth, board_t board) {
  STAT(expanded_nodes++);
  // Collect possible moves
  // board_t moves[total] = {...};
  MOVES(board)

  // Check status and transposition table for each move
  score_t best = exact_score(0);
  for (int i=total-1;i>=0;i--) {
    score_t sc = lift(quick_evaluate(moves[i]));
    if (sc>>2 >= depth)
      switch (sc&3) {
        case 2: return sc; // Win!
        case 1: best = sc; // Tie: falls through to next case
        case 0: swap(moves[i],moves[--total]); // Loss or tie: remove from the list of moves
      }
  }

  // If we're out of moves, we're done
  if (!total)
    goto done;

  // Are we out of recursion depth?
  if (depth==1) {
    best = score(1,1);
    goto done;
  }

  // No move ordering for now, since all the remaining moves are ties as far the transposition table knows
  for (int i=0;i<total;i++) {
    score_t sc = lift(evaluate_recurse(depth-1,moves[i]));
    if ((sc&3)==2)
      return sc;
    else if ((best&3)<(sc&3))
      best = sc;
  }

  // Store results and finish
  done:
  store(board,best);
  check_interrupts();
  return best;
}

// Evaluate a position down to a certain game tree depth.  If the depth is exceeded, assume a tie.
// We maintain the invariant that player 0 is always the player to move.  The result is 1 for a win
// 0 for tie, -1 for loss.
score_t evaluate(int depth, board_t board) {
  // We can afford error detection here since recursion happens into a different function
  OTHER_ASSERT(depth>=0);
  check_board(board);
  if (table_bits<10)
    throw AssertionError(format("transposition table not initialized: table_bits = %d",table_bits));

  // Exit immediately if possible
  score_t sc = quick_evaluate(board);
  if (sc>>2 >= depth)
    return sc;

  // Otherwise, recurse into children
  return evaluate_recurse(depth,board);
}

NdArray<int> unpack_py(NdArray<const board_t> boards) {
  for (int b=0;b<boards.flat.size();b++)
    check_board(boards[b]);
  Array<int> shape = boards.shape.copy();
  shape.append(6);
  shape.append(6);
  NdArray<int> tables(shape,false);
  for (int b=0;b<boards.flat.size();b++) {
    for (int qx=0;qx<2;qx++) for (int qy=0;qy<2;qy++) {
      quadrant_t q = quadrant(boards.flat[b],2*qx+qy); \
      side_t s0 = unpack(q,0), \
             s1 = unpack(q,1); \
      for (int x=0;x<3;x++) for (int y=0;y<3;y++) \
        tables.flat[36*b+6*(3*qx+x)+3*qy+y] = ((s0>>(3*x+y))&1)+2*((s1>>(3*x+y))&1); \
    }
  }
  return tables;
}

NdArray<board_t> pack_py(NdArray<const int> tables) {
  OTHER_ASSERT(tables.rank()>=2);
  int r = tables.rank();
  OTHER_ASSERT(tables.shape[r-2]==6 && tables.shape[r-1]==6);
  NdArray<board_t> boards(tables.shape.slice(0,r-2).copy(),false);
  for (int b=0;b<boards.flat.size();b++) {
    quadrant_t q[4];
    for (int qx=0;qx<2;qx++) for (int qy=0;qy<2;qy++) {
      quadrant_t s0=0,s1=0;
      for (int x=0;x<3;x++) for (int y=0;y<3;y++) {
        quadrant_t bit = 1<<(3*x+y);
        switch (tables.flat[36*b+6*(3*qx+x)+3*qy+y]) {
          case 1: s0 |= bit; break;
          case 2: s1 |= bit; break;
        }
      }
      q[2*qx+qy] = pack(s0,s1);
    }
    boards.flat[b] = quadrants(q[0],q[1],q[2],q[3]);
  }
  return boards;
}

inline board_t pack(const Vector<Vector<quadrant_t,2>,4>& sides) {
    return quadrants(pack(sides[0][0],sides[0][1]),
                     pack(sides[1][0],sides[1][1]),
                     pack(sides[2][0],sides[2][1]),
                     pack(sides[3][0],sides[3][1]));
}

// Rotate and reflect a board to minimize its value
board_t standardize(board_t board) {
  Vector<Vector<quadrant_t,2>,4> sides;
  for (int q=0;q<4;q++) for (int s=0;s<2;s++)
    sides[q][s] = unpack(quadrant(board,q),s);
  board_t transformed[8];
  for (int rotation=0;rotation<4;rotation++) {
    for (int reflection=0;reflection<2;reflection++) {
      transformed[2*rotation+reflection] = pack(sides);
      // Reflect about y axis
      for (int q=0;q<4;q++)
        for (int s=0;s<2;s++) {
          // static const quadrant_t reflections[512][2] = {...};
          #include "gen/reflect.h"
          sides[q][s] = reflections[sides[q][s]][0];
        }
      for (int qy=0;qy<2;qy++)
        swap(sides[qy],sides[2+qy]);
    }
    // Rotate left
    for (int q=0;q<4;q++)
      for (int s=0;s<2;s++)
        sides[q][s] = rotations[sides[q][s]][0];
    Vector<Vector<quadrant_t,2>,4> prev = sides;
    sides[0] = prev[1];
    sides[1] = prev[3];
    sides[2] = prev[0];
    sides[3] = prev[2];
  }
  return RawArray<board_t>(8,transformed).min();
}

NdArray<board_t> standardize_py(NdArray<const board_t> boards) {
  NdArray<board_t> transformed(boards.shape,false);
  for (int b=0;b<boards.flat.size();b++)
    transformed.flat[b] = standardize(boards.flat[b]);
  return transformed;
}

int status_py(board_t board) {
  check_board(board);
  return status(board);
}

Array<board_t> moves(board_t board) {
  check_board(board);
  // const board_t moves[total] = {...};
  MOVES(board)
  return RawArray<board_t>(total,moves).copy();
}

PyObject* stats() {
  PyObject* stats = PyDict_New();
  #define ST(stat) PyDict_SetItemString(stats,#stat,PyInt_FromLong(stat));
  ST(expanded_nodes)
  ST(total_lookups)
  ST(successful_lookups)
  return stats;
}

}

OTHER_PYTHON_MODULE(pentago) {
  using namespace python;
  function("status",status_py);
  OTHER_FUNCTION(evaluate)
  OTHER_FUNCTION(moves)
  OTHER_FUNCTION(init_table)
  OTHER_FUNCTION(clear_stats)
  OTHER_FUNCTION(stats)
  function("unpack",unpack_py);
  function("pack",pack_py);
  function("standardize",standardize_py);
}
