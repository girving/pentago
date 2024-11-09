// Board enumeration

#include "pentago/base/all_boards.h"
#include "pentago/base/symmetry.h"
#include "pentago/base/count.h"
#include "pentago/base/hash.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/large.h"
#include "pentago/utility/log.h"
#include "pentago/utility/popcount.h"
#include "pentago/utility/random.h"
#include "pentago/utility/box.h"
#include <unordered_map>
#include <unordered_set>
namespace pentago {

using std::get;
using std::make_pair;
using std::max;
using std::min;
using std::unordered_map;
using std::unordered_set;
using std::vector;

template<int symmetries> static inline board_t maybe_standardize(board_t board);
template<> board_t maybe_standardize<1>(board_t board) { return board; }
template<> board_t maybe_standardize<8>(board_t board) { return standardize(board); }
template<> board_t maybe_standardize<2048>(board_t board) { return get<0>(superstandardize(board)); }

// List all standardized boards with n stones, assuming black plays first.
// The symmetries are controls how many symmetries are taken into account: 1 for none, 8 for global, 2048 for super.
template<int symmetries> static Array<board_t> all_boards_helper(int n) {
  GEODE_ASSERT(0<=n && n<=36);
  const uint32_t batch = 1000000;
  const bool verbose = false;

  // Make a list of all single stone boards
  board_t singletons[36];
  for (int i=0;i<36;i++)
    singletons[i] = (board_t)pack_table[1<<i%9]<<16*(i/9);

  // Generate all black boards with n stones
  vector<board_t> black_boards;
  if (verbose)
    slog("black board bound = %d, bound/batch = %g", choose(36,n), double(choose(36,n))/batch);
  {
    board_t board = 0;
    uint64_t count = 0;
    int stack[36];
    unordered_set<board_t> black_board_set;
    for (int depth=0,next=0;;) {
      if (depth==n) {
        count++;
        board_t standard = maybe_standardize<symmetries>(board);
        if (black_board_set.insert(standard).second) {
          black_boards.push_back(standard);
          if (verbose && black_boards.size()%batch==0)
            slog("black boards = %d", black_boards.size());
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
    GEODE_ASSERT(count==choose(36,n));
  }

  // Generate all n/2-subsets of [0,n)
  vector<uint64_t> subsets;
  const int white = n/2;
  {
    uint64_t subset = 0;
    int stack[36];
    for (int depth=0,next=0;;) {
      if (depth==white) {
        GEODE_ASSERT(popcount(subset)==white);
        subsets.push_back(subset);
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
    GEODE_ASSERT((uint64_t)subsets.size()==choose(n,white));
  }

  // Combine black_boards and subsets to produce all boards with n stones
  vector<board_t> boards;
  if (verbose) {
    uint64_t bound = black_boards.size()*subsets.size();
    slog("board bound = %d, bound/batch = %g", bound, double(bound)/batch);
  }
  {
    unordered_set<board_t> board_set;
    for (const board_t black : black_boards) {
      board_set.clear();
      // Make a list of occupied singleton boards
      board_t occupied[n];
      int c = 0;
      for (int i=0;i<36;i++)
        if (unpack(black,0) & unpack(singletons[i], 0))
          occupied[c++] = singletons[i];
      GEODE_ASSERT(c == n);
      // Traverse all white subsets
      for (const uint64_t subset : subsets) {
        board_t board = black;
        for (int i=0;i<n;i++)
          if (subset&(uint64_t)1<<i)
            board += occupied[i];
        board = maybe_standardize<symmetries>(board);
        if (board_set.insert(board).second) {
          GEODE_ASSERT(   popcount(unpack(board,0))==n-white
                       && popcount(unpack(board,1))==white);
          boards.push_back(board);
          if (verbose && boards.size() % batch == 0)
            slog("boards = %d", boards.size());
        }
      }
    }
  }
  GEODE_ASSERT(count_boards(n, symmetries) == uint64_t(boards.size()));
  return asarray(boards).copy();
}

Array<board_t> all_boards(const int n, const int symmetries) {
  GEODE_ASSERT(symmetries==1 || symmetries==8 || symmetries==2048);
  return symmetries==1?all_boards_helper<1>(n)
        :symmetries==8?all_boards_helper<8>(n)
          /* 2048 */  :all_boards_helper<2048>(n);
}

Array<quadrant_t> minimal_quadrants() {
  vector<quadrant_t> mins;
  for (quadrant_t q=0;q<quadrant_count;q++)
    if (get<0>(superstandardize(q))==q)
      mins.push_back(q);
  return asarray(mins).copy();
}

Array<section_t> all_boards_sections(const int n, const int symmetries) {
  if (n == -1) {
    // Collect all sections
    vector<section_t> sections;
    for (const int i : range(36+1))
      extend(sections, all_boards_sections(i,symmetries));
    return asarray(sections).copy();
  }
  GEODE_ASSERT(0<=n && n<=36);
  GEODE_ASSERT(symmetries==1 || symmetries==4 || symmetries==8);
  const int white = n/2, black = n-white;
  vector<section_t> sections;

  // Loop over possible counts in quadrant 3
  for (int b3=0;b3<=min(black,9);b3++) {
    for (int w3=0;w3<=min(white,9-b3);w3++) {
      const Vector<int,2> left3(black-b3,white-w3);
      // Loop over possible counts in quadrant 2
      const Box<int> range_b2(0,min(left3[0],9));
      for (int b2=range_b2.min;b2<=range_b2.max;b2++) {
        const Box<int> range_w2(max(0,left3[1]-(18-(left3[0]-b2))),min(left3[1],9-b2));
        for (int w2=range_w2.min;w2<=range_w2.max;w2++) {
          const Vector<int,2> left2 = left3-vec(b2,w2);
          // Loop over possible counts in quadrant 1
          const Box<int> range_b1(max(0,left2[0]-9),min(left2[0],9));
          for (int b1=range_b1.min;b1<=range_b1.max;b1++) {
            const Box<int> range_w1(max(0,left2[1]-(9-left2[0]+b1)),min(left2[1],9-b1));
            for (int w1=range_w1.min;w1<=range_w1.max;w1++) {
              const Vector<int,2> left1 = left2-vec(b1,w1);
              // Quadrant 0's counts are now uniquely determined
              const int b0 = left1[0], w0 = left1[1];
              // We've found a section!
              section_t s(vec(Vector<uint8_t,2>(b0,w0),Vector<uint8_t,2>(b1,w1),Vector<uint8_t,2>(b2,w2),Vector<uint8_t,2>(b3,w3)));
              if (symmetries==1 || (symmetries==4 && get<0>(s.standardize<4>())==s) ||
                  (symmetries==8 && get<0>(s.standardize<8>())==s))
                sections.push_back(s);
            }
          }
        }
      }
    }
  }
  std::sort(sections.begin(), sections.end());
  return asarray(sections).copy();
}

Array<uint64_t> all_boards_section_sizes(int n, int symmetries) {
  vector<uint64_t> sizes;
  for (section_t s : all_boards_sections(n,symmetries))
    sizes.push_back(s.size());
  return asarray(sizes).copy();
}

uint64_t all_boards_stats(const int n, const int symmetries) {
  if (n==0) {
    RawArray<const uint16_t> offsets(10*(10+1)/2,rotation_minimal_quadrants_offsets);
    int max_bucket = 0;
    for (int i = 0; i < offsets.size()-1; i++)
      max_bucket = max(max_bucket, offsets[i+1] - offsets[i]);
    slog("maximum rmin bucket size = %d", max_bucket);
  }
  Array<section_t> sections = all_boards_sections(n,1);
  int reduced_sections = 0;
  uint64_t max_section = 0;
  Box<uint64_t> blocks;
  Box<uint64_t> lines;
  uint64_t total = 0;
  uint64_t reduced_total = 0;
  for (section_t s : sections) {
    const uint64_t size = s.size();
    max_section = max(max_section,size);
    total += size;
    if (symmetries==1 || (symmetries==4 && get<0>(s.standardize<4>())==s) ||
        (symmetries==8 && get<0>(s.standardize<8>())==s)) {
      reduced_sections++;
      const auto shape = s.shape();
      const auto lo = (shape/8).sorted(),
                 hi = ceil_div(shape,8).sorted();
      const auto lop = lo.product(),
                 hip = hi.product();
      blocks.min += lop;
      blocks.max += hip;
      lines.min += lop?lop/lo[0]+lop/lo[1]+lop/lo[2]+lop/lo[3]:0;
      lines.max += hip/hi[0]+hip/hi[1]+hip/hi[2]+hip/hi[3];
      reduced_total += size;
    }
  }
  const uint64_t exact = count_boards(n,2048);
  const double inv_exact = exact ? 1./exact : 0;
  slog("%s, simple count = %18s, ratio = %5.3f (unreduced %5.3f), sections = %6d (unreduced %6d), "
       "blocks = %10lld %10lld, lines = %9lld %9lld, max section = %14s, mean = %.4g",
    n < 0 ? "   all" : tfm::format("n = %2d",n),
    large(reduced_total), inv_exact*reduced_total, inv_exact*total, reduced_sections, sections.size(),
    blocks.min, blocks.max, lines.min, lines.max, large(max_section),
    reduced_sections ? (double)reduced_total/reduced_sections : 0);
  GEODE_ASSERT(8*reduced_sections>=sections.size());
  return reduced_total;
}

Array<board_t> all_boards_list(int n) {
  Array<section_t> sections = all_boards_sections(n);

  // Make sure we fit into Array
  uint64_t count = 0;
  for (section_t s : sections)
    count += s.size();
  CHECK_CAST_INT(count);

  // Collect boards
  vector<board_t> list;
  for (section_t s : sections) {
    RawArray<const quadrant_t> bucket0 = get<0>(rotation_minimal_quadrants(s.counts[0])),
                               bucket1 = get<0>(rotation_minimal_quadrants(s.counts[1])),
                               bucket2 = get<0>(rotation_minimal_quadrants(s.counts[2])),
                               bucket3 = get<0>(rotation_minimal_quadrants(s.counts[3]));
    for (auto q0 : bucket0)
      for (auto q1 : bucket1)
        for (auto q2 : bucket2)
          for (auto q3 : bucket3)
            list.push_back(quadrants(q0,q1,q2,q3));
  }
  return asarray(list).copy();
}

}
