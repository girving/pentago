// Board enumeration

#include "board.h"
#include "symmetry.h"
#include "table.h"
#include <other/core/array/NdArray.h>
#include <other/core/math/popcount.h>
#include <other/core/math/uint128.h>
#include <other/core/python/module.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/utility/interrupts.h>
namespace pentago {

using namespace other;
using std::cout;
using std::endl;

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

template<int symmetries> static inline board_t maybe_standardize(board_t board);
template<> inline board_t maybe_standardize<1>(board_t board) { return board; }
template<> inline board_t maybe_standardize<8>(board_t board) { return standardize(board); }
template<> inline board_t maybe_standardize<2048>(board_t board) { return superstandardize(board).x; }

// List all standardized boards with n stones, assuming black plays first.
// The symmetries are controls how many symmetries are taken into account: 1 for none, 8 for global, 2048 for super.
template<int symmetries> static Array<board_t> all_boards_helper(int n) {
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
        board_t standard = maybe_standardize<symmetries>(board);
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
    for (board_t black : black_boards) {
      board_set.delete_all_entries();
      // Make a list of occupied singleton boards    
      board_t occupied[n];
      int c = 0;
      for (int i=0;i<36;i++)
        if (unpack(black,0)&unpack(singletons[i],0))
          occupied[c++] = singletons[i];
      OTHER_ASSERT(c==n);
      // Traverse all white subsets
      for (uint64_t subset : subsets) {
        board_t board = black;
        for (int i=0;i<n;i++)
          if (subset&(uint64_t)1<<i)
            board += occupied[i];
        board = maybe_standardize<symmetries>(board);
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
  const uint64_t count = count_boards(n);
  OTHER_ASSERT((count+symmetries-1)/symmetries<=boards.size() && boards.size()<=count);
  return boards;
}

static Array<board_t> all_boards(int n, int symmetries) {
  OTHER_ASSERT(symmetries==1 || symmetries==8 || symmetries==2048);
  return symmetries==1?all_boards_helper<1>(n)
        :symmetries==8?all_boards_helper<8>(n)
          /* 2048 */  :all_boards_helper<2048>(n);
}

// Determine how many bits are needed to distinguish all the given boards
static int distinguishing_hash_bits(RawArray<const board_t> boards) {
  for (int bits=0;bits<=64;bits++) {
    const uint64_t mask = bits==64?(uint64_t)-1:((uint64_t)1<<bits)-1;
    Hashtable<uint64_t> hashes;
    for (board_t board : boards)
      if (!hashes.set(hash_board(board)&mask))
        goto collision; 
    return bits;
    collision:;
  }
  throw ValueError("distinguishing_hash_bits: the set of boards contains duplicates");
}

}
using namespace pentago;
using namespace other::python;

void wrap_all_boards() {
  OTHER_FUNCTION(all_boards)
  OTHER_FUNCTION(distinguishing_hash_bits)
}
