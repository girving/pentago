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
#include "gen/tables.h"
namespace pentago {

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
uint64_t distance_prunes;

void clear_stats() {
  expanded_nodes = 0;
  total_lookups = 0;
  successful_lookups = 0;
  distance_prunes = 0;
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

inline int popcount(uint64_t n) {
  BOOST_STATIC_ASSERT(sizeof(long)==sizeof(uint64_t));
  return __builtin_popcountl(n);
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

  uint64_t c = win_contributions[0][quadrant(side,0)]
             + win_contributions[1][quadrant(side,1)]
             + win_contributions[2][quadrant(side,2)]
             + win_contributions[3][quadrant(side,3)];
  return c&(c>>1)&0x55 // The first four ways of winning require contributions from three quadrants
      || c&(0xaaaaaaaaaaaaaaaa<<8); // The remaining 28 ways require contributions from only two
}

// Determine if one side can win by rotating a quadrant
inline bool rotated_won(side_t side) {
  quadrant_t q0 = quadrant(side,0),
             q1 = quadrant(side,1),
             q2 = quadrant(side,2),
             q3 = quadrant(side,3);
  // First see how far we get without rotations
  uint64_t c = win_contributions[0][q0]
             + win_contributions[1][q1]
             + win_contributions[2][q2]
             + win_contributions[3][q3];
  // We win if we only need to rotate a single quadrant
  uint64_t c0 = c+rotated_win_contribution_deltas[0][q0],
           c1 = c+rotated_win_contribution_deltas[1][q1],
           c2 = c+rotated_win_contribution_deltas[2][q2],
           c3 = c+rotated_win_contribution_deltas[3][q3];
  // Check if we won
  return (c0|c1|c2|c3)&(0xaaaaaaaaaaaaaaaa<<8) // The last remaining 28 ways of winning require contributions two quadrants
      || ((c0&(c0>>1))|(c1&(c1>>1))|(c2&(c2>>1))|(c3&(c3>>1)))&0x55; // The first four require contributions from three
}

inline quadrant_t pack(quadrant_t side0, quadrant_t side1) {
  return pack_table[side0]+2*pack_table[side1];
}

inline board_t pack(side_t side0, side_t side1) {
  return quadrants(pack(quadrant(side0,0),quadrant(side1,0)),
                   pack(quadrant(side0,1),quadrant(side1,1)),
                   pack(quadrant(side0,2),quadrant(side1,2)),
                   pack(quadrant(side0,3),quadrant(side1,3)));
}

inline quadrant_t unpack(quadrant_t state, int s) {
  assert(0<=s && s<2);
  return unpack_table[state][s];
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
enum table_type_t {blank_table,normal_table,simple_table} table_type;

void init_table(int bits) {
  if (bits<1 || bits>30)
    throw ValueError(format("expected 1<=bits<=30, got bits = %d",bits));
  if (64-bits+10>64)
    throw ValueError(format("bits = %d is too small, the high order hash bits won't fit",bits));
  free(table);
  table_bits = bits;
  table_mask = (1<<table_bits)-1;
  table = (uint64_t*)calloc(1L<<bits,sizeof(uint64_t));
  table_type = blank_table;
}

score_t lookup(board_t board) {
  STAT(total_lookups++);
  uint64_t h = hash(board);
  uint64_t entry = table[h&table_mask];
  if (entry>>score_bits==h>>table_bits) {
    STAT(successful_lookups++);
    return entry&score_mask;
  }
  return score(0,1);
}

void store(board_t board, score_t score) {
  uint64_t h = hash(board);
  uint64_t& entry = table[h&table_mask];
  if (entry>>score_bits==h>>table_bits || uint16_t(entry&score_mask)>>2 <= score>>2)
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

// We declare the move listing code as a huge macro in order to use it in multiple functions while taking advantage of gcc's variable size arrays.
// To use, invoke MOVES(board) to fill a board_t moves[total] array.
#define QR0(s,q,dir) rotations[quadrant(side##s,q)][dir]
#define QR1(s,q) {QR0(s,q,0),QR0(s,q,1)}
#define QR2(s) {QR1(s,0),QR1(s,1),QR1(s,2),QR1(s,3)}
#define COUNT_MOVES(stride,q,qpp) \
  const int offset##q  = move_offsets[quadrant(filled,q)]; \
  const int count##q   = move_offsets[quadrant(filled,q)+1]-offset##q; \
  const int total##qpp = total##q + stride*count##q;
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
  COUNT_MOVES(8,0,1) COUNT_MOVES(8,1,2) COUNT_MOVES(8,2,3) COUNT_MOVES(8,3,4) \
  int total = total4; /* Leave mutable to allow in-place pruning of the list */ \
  /* Collect the list of all possible moves.  Note that we repack with the sides */ \
  /* flipped so that it's still player 0's turn. */ \
  board_t moves[total]; \
  COLLECT_MOVES(0) COLLECT_MOVES(1) COLLECT_MOVES(2) COLLECT_MOVES(3)

// Same as MOVES, but ignores rotations and operates in unpacked mode
#define SIMPLE_COLLECT_MOVES(q) \
  for (int i=0;i<count##q;i++) \
    moves[total##q+i] = side0|(side_t)move_flat[offset##q+i]<<16*q;
#define SIMPLE_MOVES(side0,side1) \
  /* Count the number of moves in each quadrant */ \
  const side_t filled = side0|side1; \
  const int total0 = 0; \
  COUNT_MOVES(1,0,1) COUNT_MOVES(1,1,2) COUNT_MOVES(1,2,3) COUNT_MOVES(1,3,4) \
  int total = total4; /* Leave mutable to allow in-place pruning of the list */ \
  /* Collect the list of possible moves.  Note that only side0 changes */ \
  side_t moves[total]; \
  SIMPLE_COLLECT_MOVES(0) SIMPLE_COLLECT_MOVES(1) SIMPLE_COLLECT_MOVES(2) SIMPLE_COLLECT_MOVES(3)

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
  score_t best = score(depth,0);
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
    if ((sc&3)==2) {
      best = sc;
      goto done;
    } else if ((best&3)<(sc&3))
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
  if (table_type==blank_table)
    table_type = normal_table;
  OTHER_ASSERT(table_type==normal_table);

  // Exit immediately if possible
  score_t sc = quick_evaluate(board);
  if (sc>>2 >= depth)
    return sc;

  // Otherwise, recurse into children
  return evaluate_recurse(depth,board);
}

// Evaluate position based only on current state and transposition table, ignoring rotations or white five-in-a-row
template<bool black> inline score_t simple_quick_evaluate(side_t side0, side_t side1) {
  // If we're white, check if we've lost
  if (!black && rotated_won(side1))
    return exact_score(0);
  // Did we already compute this value?
  return lookup(pack(side0,side1));
}

// Compute the minimum number of black moves required for a win, together with the number of different
// ways that minimum can be achieved.  Returns ((6-min_distance)<<16)+count, so that a higher number means
// closer to a black win.  If winning is impossible, the return value is 0.
int simple_win_closeness(side_t black, side_t white) {
  // Transpose into packed quadrants
  const quadrant_t q0 = pack(quadrant(black,0),quadrant(white,0)),
                   q1 = pack(quadrant(black,1),quadrant(white,1)),
                   q2 = pack(quadrant(black,2),quadrant(white,2)),
                   q3 = pack(quadrant(black,3),quadrant(white,3));
  // Compute all distances
  const uint64_t each = 321685687669321; // sum_{i<17} 1<<3*i
  #define DISTANCES(i) ({ \
    const uint64_t d0 = simple_win_distances[0][q0][i], \
                   d1 = simple_win_distances[1][q1][i], \
                   d2 = simple_win_distances[2][q2][i], \
                   d3 = simple_win_distances[3][q3][i], \
                   blocks = (d0|d1|d2|d3)&4*each, \
                   blocked = blocks|blocks>>1|blocks>>2, \
                   unblocked = ~blocked; \
    (d0&unblocked)+(d1&unblocked)+(d2&unblocked)+(d3&unblocked)+(6*each&blocked); })
  const uint64_t d0 = DISTANCES(0), // Each of these contains 17 3-bit distances in [0,6]
                 d1 = DISTANCES(1),
                 d2 = DISTANCES(2),
                 d3 = DISTANCES(3);
  #undef DISTANCES
  // Determine minimum distance
  const bool min_under_1 = (~((d0|d0>>1|d0>>2)&(d1|d1>>1|d1>>2)&(d2|d2>>1|d2>>2)&(d3|d3>>1|d3>>2))&each)!=0, // abc < 1 iff ~(a|b|c)
             min_under_2 = (~((d0|d0>>1)&(d1|d1>>1)&(d2|d2>>1)&(d3|d3>>1))&2*each)!=0, // abc < 2 iff ~a&~b = ~(a|b)
             min_under_3 = (~((d0>>2|(d0>>1&d0))&(d1>>2|(d1>>1&d1))&(d2>>2|(d2>>1&d2))&(d3>>2|(d3>>1&d3)))&each)!=0, // abc < 3 iff ~a&(~b|~c) = ~a&~(b&c) = ~(a|(b&c))
             min_under_4 = (~(d0&d1&d2&d3)&4*each)!=0, // abc < 4 iff ~a
             min_under_5 = (~((d0>>2&(d0>>1|d0))&(d1>>2&(d1>>1|d1))&(d2>>2&(d2>>1|d2))&(d3>>2&(d3>>1|d3)))&each)!=0, // abc < 5 iff ~a|(~b&~c) = ~a|~(b|c) = ~(a&(b|c))
             min_under_6 = (~((d0&d0>>1)&(d1&d1>>1)&(d2&d2>>1)&(d3&d3>>1))&2*each)!=0; // abc < 6 iff ~a|~b = ~(a&b)
  const int min_distance = min_under_4
                             ?min_under_2
                               ?min_under_1?0:1
                               :min_under_3?2:3
                             :min_under_5
                               ?4
                               :min_under_6?5:6;
  // If we're in debug mode, check against the slow way
#ifndef NDEBUG
  #define SLOW_MIN_DISTANCE(d) ({ \
    int md = 6; \
    for (int i=0;i<17;i++) \
      md = min(md,int(d>>3*i)&7); \
    md; })
  const int slow_min_distance = min(SLOW_MIN_DISTANCE(d0),SLOW_MIN_DISTANCE(d1),SLOW_MIN_DISTANCE(d2),SLOW_MIN_DISTANCE(d3));
  OTHER_ASSERT(slow_min_distance==min_distance);
#endif
  // If the minimum distance is 6, a black win is impossible, so no need to count the ways
  if (min_distance==6)
    return 0;
  // Count number of times min_distance occurs
  const uint64_t mins = min_distance*each;
  #define MATCHES(d) (~((d^mins)|(d^mins)>>1|(d^mins)>>2)&each)
  const int count = popcount(MATCHES(d0))+popcount(MATCHES(d1)|MATCHES(d2)<<1|MATCHES(d3)<<2);
  #undef MATCHES
  return ((6-min_distance)<<16)+count;
}

// Evaluate position ignoring rotations and white five-in-a-row
template<bool black> score_t simple_evaluate_recurse(int depth, side_t side0, side_t side1) {
  STAT(expanded_nodes++);

  // Collect possible moves
  SIMPLE_MOVES(side0,side1)

  // Check status and transposition table for each move
  score_t best = black?score(depth,1):exact_score(0);
  for (int i=total-1;i>=0;i--) {
    score_t sc = lift(simple_quick_evaluate<!black>(side1,moves[i]));
    if (sc>>2 >= depth) {
      if ((sc&3)>=(black?2:1))
        return sc; // Win for black, or tie for white
      else // Tie for black, or loss for white: remove from the list of moves
        swap(moves[i],moves[--total]);
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

  // Compute how close we are to a black win
  int order[total]; // We'll sort moves in ascending order of this array
  for (int i=total-1;i>=0;i--) {
    int closeness = black?simple_win_closeness(moves[i],side1)
                         :simple_win_closeness(side1,moves[i]);
    int distance = 6-(closeness>>16);
    if (distance==6 || distance>(depth-black)/2) { // We can't reach a black win within the given search depth
      if (!black) {
        STAT(distance_prunes++);
        best = score(distance==6?36:2*distance-1+black,1);
        goto done;
      } else {
        swap(moves[i],moves[total-1]);
        swap(order[i],order[--total]);
      }
    }
    order[i] = black?-closeness:closeness;
  }

  // Insertion sort moves based on order
  for (int i=1;i<total;i++) {
    const side_t move = moves[i];
    const int key = order[i];
    int j = i-1;
    while (j>=0 && order[j]>key) {
      moves[j+1] = moves[j];
      order[j+1] = order[j];
      j--;
    }
    moves[j+1] = move;
    order[j+1] = key;
  }

  // Optionally print out move ordering information
  if (0) {
    cout << (black?"order black =":"order white =");
    for (int i=0;i<total;i++) {
      int o = black?-order[i]:order[i];
      cout << ' '<<(6-(o>>16))<<','<<(o&0xffff);
    }
    cout<<endl;
  }

  // Recurse
  for (int i=0;i<total;i++) {
    score_t sc = lift(simple_evaluate_recurse<!black>(depth-1,side1,moves[i]));
    if ((sc&3)>=(black?2:1)) {
      best = sc;
      goto done;
    }
  }

  // Store results and finish
  done:
  store(pack(side0,side1),best);
  check_interrupts();
  return best;
}

bool black_to_move(board_t board) {
  check_board(board);
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const int count0 = popcount(side0),
            count1 = popcount(side1);
  OTHER_ASSERT(count0==count1 || count1==count0+1);
  return count0==count1;
}

// Evaluate a position down to a certain game tree depth, ignoring rotations and white wins.
score_t simple_evaluate(int depth, board_t board) {
  // We can afford error detection here since recursion happens into a different function
  OTHER_ASSERT(depth>=0);
  check_board(board);
  if (table_bits<10)
    throw AssertionError(format("transposition table not initialized: table_bits = %d",table_bits));
  if (table_type==blank_table)
    table_type = simple_table;
  OTHER_ASSERT(table_type==simple_table);

  // Determine whether player 0 is black (first to move) or white (second to move)
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  bool black = black_to_move(board);

  // Exit immediately if possible
  score_t sc = black?simple_quick_evaluate<true>(side0,side1)
                    :simple_quick_evaluate<false>(side0,side1);
  if (sc>>2 >= depth)
    return sc;

  // Otherwise, recurse into children
  return black?simple_evaluate_recurse<true >(depth,side0,side1)
              :simple_evaluate_recurse<false>(depth,side0,side1);
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

int simple_status(board_t board) {
  check_board(board);
  return rotated_won(unpack(board,0))?1:0;
}

Array<board_t> moves(board_t board) {
  check_board(board);
  // const board_t moves[total] = {...};
  MOVES(board)
  return RawArray<board_t>(total,moves).copy();
}

Array<board_t> simple_moves(board_t board) {
  check_board(board);
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  // const side_t moves[total] = {...};
  SIMPLE_MOVES(side0,side1)
  Array<board_t> result(total,false);
  for (int i=0;i<total;i++)
    result[i] = pack(side1,moves[i]);
  return result;
}

PyObject* stats() {
  PyObject* stats = PyDict_New();
  #define ST(stat) PyDict_SetItemString(stats,#stat,PyInt_FromLong(stat));
  ST(expanded_nodes)
  ST(total_lookups)
  ST(successful_lookups)
  ST(distance_prunes)
  return stats;
}

}
}
using namespace pentago;

OTHER_PYTHON_MODULE(pentago) {
  using namespace python;
  function("status",status_py);
  OTHER_FUNCTION(evaluate)
  OTHER_FUNCTION(simple_evaluate)
  OTHER_FUNCTION(moves)
  OTHER_FUNCTION(simple_moves)
  OTHER_FUNCTION(init_table)
  OTHER_FUNCTION(clear_stats)
  OTHER_FUNCTION(stats)
  OTHER_FUNCTION(black_to_move)
  OTHER_FUNCTION(simple_status)
  function("unpack",unpack_py);
  function("pack",pack_py);
  function("standardize",standardize_py);
}
