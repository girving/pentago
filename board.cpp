// Board definitions and utility functions

#include "board.h"
#include <other/core/array/NdArray.h>
#include <other/core/math/popcount.h>
#include <other/core/math/uint128.h>
#include <other/core/python/module.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/utility/foreach.h>
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
  OTHER_ASSERT(count0==count1 || count0==count1+1);
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

static NdArray<board_t> standardize_py(NdArray<const board_t> boards) {
  NdArray<board_t> transformed(boards.shape,false);
  for (int b=0;b<boards.flat.size();b++)
    transformed.flat[b] = standardize(boards.flat[b]);
  return transformed;
}

static uint64_t safe_mul(uint64_t a, uint64_t b) {
  uint128_t ab = uint128_t(a)*b;
  OTHER_ASSERT(!(ab>>64));
  return uint64_t(ab);
}

static uint64_t choose(int n, int k) {
  if (n<0 || k<0 || k>n)
    return 0;
  k = max(k,n-k);
  uint64_t result = 1;
  for (int i=n;i>k;i--)
    result = safe_mul(result,i);
  for (int i=2;i<=n-k;i++)
    result /= i;
  return result;
}

// List all unstandardized boards with n stones, assuming black plays first
static uint64_t count_boards(int n) {
  OTHER_ASSERT(0<=n && n<=36);
  return choose(36,n)*choose(n,n/2);
}

// List all standardized boards with n stones, assuming black plays first
static Array<board_t> all_boards(int n) {
  OTHER_ASSERT(0<=n && n<=36);
  const uint32_t batch = 1000000;
  const bool verbose = false;

  // Make a list of all single stone boards
  board_t singletons[36];
  for (int i=0;i<36;i++)
    singletons[i] = (board_t)pack_table[1<<i%9]<<16*(i/9);

  // Generate all black boards with n stones
  Array<board_t> black_boards;
  if (verbose)
    cout << "black board bound = "<<choose(36,n)<<", bound/batch = "<<double(choose(36,n))/batch<<endl;
  {
    board_t board = 0;
    uint64_t count = 0;
    int stack[36];
    Hashtable<board_t> black_board_set;
    for (int depth=0,next=0;;) {
      if (depth==n) {
        count++;
        board_t standard = standardize(board);
        if (black_board_set.set(standard)) {
          black_boards.append(standard);
          if (verbose && black_boards.size()%batch==0)
            cout << "black boards = "<<black_boards.size()<<endl;
          check_interrupts();
        }
        goto pop0;
      } else if (next-depth>36-n)
        goto pop0;
      // Recurse downwards
      board += singletons[next];
      stack[depth++] = next++;
      continue;
      // Return upwards
      pop0:
      if (!depth--)
        break;
      next = stack[depth];
      board -= singletons[next];
      next++;
    }
    OTHER_ASSERT(count==choose(36,n));
  }

  // Generate all n/2-subsets of [0,n)
  Array<uint64_t> subsets;
  const int white = n/2;
  {
    uint64_t subset = 0;
    int stack[36];
    for (int depth=0,next=0;;) {
      if (depth==white) {
        OTHER_ASSERT(popcount(subset)==white);
        subsets.append(subset);
        check_interrupts();
        goto pop1;
      } else if (next-depth>n-white)
        goto pop1;
      // Recurse downwards
      subset |= 1<<next;
      stack[depth++] = next++;
      continue;
      // Return upwards
      pop1:
      if (!depth--)
        break;
      next = stack[depth];
      subset -= 1<<next;
      next++;
    }
    OTHER_ASSERT((uint64_t)subsets.size()==choose(n,white));
  }

  // Combine black_boards and subsets to produce all boards with n stones
  Array<board_t> boards;
  if (verbose) {
    uint64_t bound = black_boards.size()*subsets.size();
    cout << "board bound = "<<bound<<", bound/batch = "<<double(bound)/batch<<endl;
  }
  {
    Hashtable<board_t> board_set;
    foreach (board_t black, black_boards) {
      board_set.delete_all_entries();
      // Make a list of occupied singleton boards    
      board_t occupied[n];
      int c = 0;
      for (int i=0;i<36;i++)
        if (unpack(black,0)&unpack(singletons[i],0))
          occupied[c++] = singletons[i];
      OTHER_ASSERT(c==n);
      // Traverse all white subsets
      foreach (uint64_t subset, subsets) {
        board_t board = black;
        for (int i=0;i<n;i++)
          if (subset&(uint64_t)1<<i)
            board += occupied[i];
        board = standardize(board);
        if (board_set.set(board)) {
          OTHER_ASSERT(   popcount(unpack(board,0))==n-white
                       && popcount(unpack(board,1))==white);
          boards.append(board);
          if (verbose && boards.size()%batch==0)
            cout << "boards = "<<boards.size()<<endl;
          check_interrupts();
        }
      }
    }
  }
  const int count = count_boards(n);
  OTHER_ASSERT((count+7)/8<=boards.size() && boards.size()<=count);
  return boards;
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
  OTHER_FUNCTION(all_boards)
}
