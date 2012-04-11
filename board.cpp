// Board definitions and utility functions

#include "board.h"
#include <other/core/array/NdArray.h>
#include <other/core/math/popcount.h>
#include <other/core/python/module.h>
#include <other/core/random/Random.h>
#include <other/core/utility/format.h>
#include <other/core/utility/interrupts.h>
namespace pentago {

using namespace other;
using std::cout;
using std::endl;

bool black_to_move(board_t board) {
  check_board(board);
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const int count0 = popcount(side0),
            count1 = popcount(side1);
  OTHER_ASSERT(count0==count1 || count1==count0+1);
  return count0==count1;
}

void check_board(board_t board) {
  #define CHECK(q) \
    if (!(quadrant(board,q)<(int)pow(3.,9.))) \
      throw ValueError(format("quadrant %d has invalid value %d",q,quadrant(board,q)));
  CHECK(0) CHECK(1) CHECK(2) CHECK(3)
}

static NdArray<int> unpack_py(NdArray<const board_t> boards) {
  for (int b=0;b<boards.flat.size();b++)
    check_board(boards.flat[b]);
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

static NdArray<board_t> pack_py(NdArray<const int> tables) {
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

static NdArray<board_t> standardize_py(NdArray<const board_t> boards) {
  NdArray<board_t> transformed(boards.shape,false);
  for (int b=0;b<boards.flat.size();b++)
    transformed.flat[b] = standardize(boards.flat[b]);
  return transformed;
}

side_t random_side(Random& random) {
  return random.bits<uint64_t>()&side_mask;
}

board_t random_board(Random& random) {
  side_t filled = random_side(random);
  side_t black = random.bits<uint64_t>()&filled;
  return pack(black,filled^black);
}

}
using namespace pentago;
using namespace other::python;

void wrap_board() {
  function("unpack",unpack_py);
  function("pack",pack_py);
  function("standardize",standardize_py);
  OTHER_FUNCTION(check_board)
  OTHER_FUNCTION(black_to_move)
}
