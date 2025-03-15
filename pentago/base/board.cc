// Board definitions and utility functions

#include "pentago/base/board.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/array.h"
#include "pentago/utility/popcount.h"
#include "pentago/utility/range.h"
#include "pentago/utility/format.h"
#include "pentago/utility/random.h"
namespace pentago {

using std::min;
using std::swap;

bool black_to_move(board_t board) {
  check_board(board);
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const int count0 = popcount(side0),
            count1 = popcount(side1);
  GEODE_ASSERT(count0==count1 || count1==count0+1);
  return count0==count1;
}

void check_board(board_t board) {
  #define CHECK(q) \
    if (!(quadrant(board, q) < 19683)) \
      THROW(ValueError, "quadrant %d has invalid value %d", q, quadrant(board, q));
  CHECK(0) CHECK(1) CHECK(2) CHECK(3)
}

static inline board_t pack(const Vector<Vector<quadrant_t,2>,4>& sides) {
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
      // Reflect about x = y line
      for (int q=0;q<4;q++)
        for (int s=0;s<2;s++)
          sides[q][s] = reflections[sides[q][s]];
      swap(sides[0],sides[3]);
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

Array<int,2> to_table(const board_t board) {
  check_board(board);
  Array<int,2> table(6, 6, uninit);
  for (const int qx : range(2)) {
    for (const int qy : range(2)) {
      const quadrant_t q = quadrant(board, 2*qx+qy);
      const side_t s0 = unpack(q,0),
                   s1 = unpack(q,1);
      for (const int x : range(3))
        for (const int y : range(3))
          table(3*qx+x, 3*qy+y) = ((s0>>(3*x+y))&1)+2*((s1>>(3*x+y))&1);
    }
  }
  return table;
}

board_t from_table(RawArray<const int,2> table) {
  GEODE_ASSERT(table.shape() == vec(6,6));
  quadrant_t q[4];
  for (const int qx : range(2)) {
    for (const int qy : range(2)) {
      quadrant_t s0 = 0, s1 = 0;
      for (const int x : range(3)) {
        for (const int y : range(3)) {
          const quadrant_t bit = 1<<(3*x+y);
          switch (table(3*qx+x, 3*qy+y)) {
            case 1: s0 |= bit; break;
            case 2: s1 |= bit; break;
          }
        }
      }
      q[2*qx+qy] = pack(s0, s1);
    }
  }
  return quadrants(q[0], q[1], q[2], q[3]);
}

side_t random_side(Random& random) {
  return random.bits<uint64_t>()&side_mask;
}

board_t random_board(Random& random) {
  side_t filled = random_side(random);
  side_t black = random.bits<uint64_t>() & filled;
  return pack(black, filled ^ black);
}

static side_t add_random_stone(Random& random, side_t filled) {
  for (;;) {
    int j = random.uniform<int>(0,36);
    side_t stone = (side_t)1<<(16*(j&3)+j/4);
    if (!(stone&filled)) {
      GEODE_ASSERT(popcount(filled)+1==popcount(stone|filled));
      return stone|filled;
    }
  }
}

// Generate a random board with n stones
board_t random_board(Random& random, int n) {
  const int nw = n/2, nb = n-nw, ne = 36-n;
  // Randomly place white stones
  side_t white = 0;
  for (int i=0;i<nw;i++)
    white = add_random_stone(random,white);
  // Random place either black stones or empty
  const int no = min(nb,ne);
  side_t other = white;
  for (int i=0;i<no;i++)
    other = add_random_stone(random,other);
  // Construct board
  const side_t black = no==nb?other&~white:side_mask&~other;
  GEODE_ASSERT(popcount(white)==nw);
  GEODE_ASSERT(popcount(black)==nb);
  GEODE_ASSERT(popcount(white|black)==n);
  return pack(black,white);
}

string str_board(board_t board) {
  string s;
  s += tfm::format("counts: 0s = %d, 1s = %d\n\n",popcount(unpack(board,0)),popcount(unpack(board,1)));
  const Array<const int,2> table = to_table(board);
  for (int i=0;i<6;i++) {
    int y = 5-i;
    s += "abcdef"[i];
    s += "  ";
    for (int x=0;x<6;x++)
      s += "_01"[table(x,y)];
    s += '\n';
  }
  return s+"\n   123456";
}

}  // namespace pentago
