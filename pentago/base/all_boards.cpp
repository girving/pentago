// Board enumeration

#include <pentago/base/all_boards.h>
#include <pentago/base/symmetry.h>
#include <pentago/base/count.h>
#include <pentago/base/hash.h>
#include <pentago/utility/ceil_div.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/large.h>
#include <geode/array/sort.h>
#include <geode/math/popcount.h>
#include <geode/python/Class.h>
#include <geode/python/wrap.h>
#include <geode/random/Random.h>
#include <geode/structure/Hashtable.h>
#include <geode/utility/interrupts.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/str.h>
#include <geode/geometry/Box.h>
namespace pentago {

using namespace geode;
using std::cout;
using std::endl;
using std::vector;

template<int symmetries> static inline board_t maybe_standardize(board_t board);
template<> inline board_t maybe_standardize<1>(board_t board) { return board; }
template<> inline board_t maybe_standardize<8>(board_t board) { return standardize(board); }
template<> inline board_t maybe_standardize<2048>(board_t board) { return superstandardize(board).x; }

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
    GEODE_ASSERT(count==choose(36,n));
  }

  // Generate all n/2-subsets of [0,n)
  Array<uint64_t> subsets;
  const int white = n/2;
  {
    uint64_t subset = 0;
    int stack[36];
    for (int depth=0,next=0;;) {
      if (depth==white) {
        GEODE_ASSERT(popcount(subset)==white);
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
    GEODE_ASSERT((uint64_t)subsets.size()==choose(n,white));
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
      board_set.clear();
      // Make a list of occupied singleton boards
      board_t occupied[n];
      int c = 0;
      for (int i=0;i<36;i++)
        if (unpack(black,0)&unpack(singletons[i],0))
          occupied[c++] = singletons[i];
      GEODE_ASSERT(c==n);
      // Traverse all white subsets
      for (uint64_t subset : subsets) {
        board_t board = black;
        for (int i=0;i<n;i++)
          if (subset&(uint64_t)1<<i)
            board += occupied[i];
        board = maybe_standardize<symmetries>(board);
        if (board_set.set(board)) {
          GEODE_ASSERT(   popcount(unpack(board,0))==n-white
                       && popcount(unpack(board,1))==white);
          boards.append(board);
          if (verbose && boards.size()%batch==0)
            cout << "boards = "<<boards.size()<<endl;
          check_interrupts();
        }
      }
    }
  }
  GEODE_ASSERT(count_boards(n,symmetries)==(uint64_t)boards.size());
  return boards;
}

static Array<board_t> all_boards(int n, int symmetries) {
  GEODE_ASSERT(symmetries==1 || symmetries==8 || symmetries==2048);
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
  THROW(ValueError,"distinguishing_hash_bits: the set of boards contains duplicates");
}

// Quadrants minimal w.r.t. rotations and reflections
static Array<quadrant_t> minimal_quadrants() {
  Array<quadrant_t> mins;
  for (quadrant_t q=0;q<quadrant_count;q++)
    if (superstandardize(q).x==q)
      mins.append(q);
  return mins;
}

Array<section_t> all_boards_sections(const int n, const int symmetries) {
  if (n == -1) {
    // Collect all sections
    Array<section_t> sections;
    for (const int i : range(36+1))
      sections.extend(all_boards_sections(i,symmetries));
    return sections;
  }
  GEODE_ASSERT(0<=n && n<=36);
  GEODE_ASSERT(symmetries==1 || symmetries==4 || symmetries==8);
  const int white = n/2, black = n-white;
  Array<section_t> sections;

  // Loop over possible counts in quadrant 3
  for (int b3=0;b3<=min(black,9);b3++) {
    for (int w3=0;w3<=min(white,9-b3);w3++) {
      const Vector<int,2> left3(black-b3,white-w3);
      // Loop over possible counts in quadrant 2
      const Box<int> range_b2(0,min(left3.x,9));
      for (int b2=range_b2.min;b2<=range_b2.max;b2++) {
        const Box<int> range_w2(max(0,left3.y-(18-(left3.x-b2))),min(left3.y,9-b2));
        for (int w2=range_w2.min;w2<=range_w2.max;w2++) {
          const Vector<int,2> left2 = left3-vec(b2,w2);
          // Loop over possible counts in quadrant 1
          const Box<int> range_b1(max(0,left2.x-9),min(left2.x,9));
          for (int b1=range_b1.min;b1<=range_b1.max;b1++) {
            const Box<int> range_w1(max(0,left2.y-(9-left2.x+b1)),min(left2.y,9-b1));
            for (int w1=range_w1.min;w1<=range_w1.max;w1++) {
              const Vector<int,2> left1 = left2-vec(b1,w1);
              // Quadrant 0's counts are now uniquely determined
              const int b0 = left1.x, w0 = left1.y;
              // We've found a section!
              section_t s(vec(Vector<uint8_t,2>(b0,w0),Vector<uint8_t,2>(b1,w1),Vector<uint8_t,2>(b2,w2),Vector<uint8_t,2>(b3,w3)));
              if (symmetries==1 || (symmetries==4 && s.standardize<4>().x==s) || (symmetries==8 && s.standardize<8>().x==s))
                sections.append(s);
            }
          }
        }
      }
    }
  }
  sort(sections);
  return sections;
}

Array<uint64_t> all_boards_section_sizes(int n, int symmetries) {
  Array<uint64_t> sizes;
  for (section_t s : all_boards_sections(n,symmetries))
    sizes.append(s.size());
  return sizes;
}

uint64_t all_boards_stats(const int n, const int symmetries) {
  if (n==0) {
    RawArray<const uint16_t> offsets(10*(10+1)/2,rotation_minimal_quadrants_offsets);
    cout << "maximum rmin bucket size = "<<(offsets.slice(1,offsets.size())-offsets.slice(0,offsets.size()-1)).max()<<endl;
  }
  Array<section_t> sections = all_boards_sections(n,1);
  int reduced_sections = 0;
  uint64_t max_section = 0;
  Box<uint64_t> blocks = 0;
  Box<uint64_t> lines = 0;
  uint64_t total = 0;
  uint64_t reduced_total = 0;
  for (section_t s : sections) {
    const uint64_t size = s.size();
    max_section = max(max_section,size);
    total += size;
    if (symmetries==1 || (symmetries==4 && s.standardize<4>().x==s) || (symmetries==8 && s.standardize<8>().x==s)) {
      reduced_sections++;
      const auto shape = s.shape();
      const auto lo = (shape/8).sorted(),
                 hi = ceil_div(shape,8).sorted();
      const auto lop = lo.product(),
                 hip = hi.product();
      blocks.min += lop;
      blocks.max += hip;
      lines.min += lop?lop/lo.x+lop/lo.y+lop/lo.z+lop/lo.w:0;
      lines.max += hip/hi.x+hip/hi.y+hip/hi.z+hip/hi.w;
      reduced_total += size;
    }
  }
  const uint64_t exact = count_boards(n,2048);
  const double inv_exact = exact ? 1./exact : 0;
  cout << format("%s, simple count = %18s, ratio = %5.3f (unreduced %5.3f), sections = %6d (unreduced %6d), blocks = %10lld %10lld, lines = %9lld %9lld, max section = %14s, mean = %.4g",
    n < 0 ? "   all" : format("n = %2d",n),
    large(reduced_total),inv_exact*reduced_total,inv_exact*total,reduced_sections,sections.size(),blocks.min,blocks.max,lines.min,lines.max,large(max_section),
    reduced_sections?(double)reduced_total/reduced_sections:0) << endl;
  GEODE_ASSERT(8*reduced_sections>=sections.size());
  return reduced_total;
}

Array<board_t> all_boards_list(int n) {
  Array<section_t> sections = all_boards_sections(n);

  // Make sure we fit into Array
  uint64_t large_count = 0;
  for (section_t s : sections)
    large_count += s.size();
  const int small_count = CHECK_CAST_INT(large_count);

  // Collect boards
  Array<board_t> list;
  list.preallocate(small_count);
  for (section_t s : sections) {
    check_interrupts();
    RawArray<const quadrant_t> bucket0 = rotation_minimal_quadrants(s.counts[0]).x,
                               bucket1 = rotation_minimal_quadrants(s.counts[1]).x,
                               bucket2 = rotation_minimal_quadrants(s.counts[2]).x,
                               bucket3 = rotation_minimal_quadrants(s.counts[3]).x;
    for (auto q0 : bucket0)
      for (auto q1 : bucket1)
        for (auto q2 : bucket2)
          for (auto q3 : bucket3)
            list.append_assuming_enough_space(quadrants(q0,q1,q2,q3));
  }
  return list;
}

void all_boards_sample_test(int n, int steps) {
  Array<section_t> sections = all_boards_sections(n);

  // Sort buckets in preparation for binary search
  Hashtable<Vector<uint8_t,2>,Array<quadrant_t>> sorted;
  for (uint8_t b=0;b<=9;b++)
    for (uint8_t w=0;w<=9-b;w++) {
      auto bucket = rotation_minimal_quadrants(b,w).x.copy();
      sort(bucket);
      sorted.set(vec(b,w),bucket);
    }

  // Generate a bunch of random boards, and check that each one occurs in a section
  Ref<Random> random = new_<Random>(175131);
  for (int step=0;step<steps;step++) {
    const board_t board = random_board(random,n);
    const section_t s = count(board);
    section_t ss;uint8_t g;s.standardize<8>().get(ss,g);
    // Does this section exist?
    const int si = int(std::lower_bound(sections.begin(),sections.end(),ss)-sections.begin());
    if (!(sections.valid(si) && sections[si]==ss)) {
      cout << "missing section "<<s.counts<<", standard "<<ss.counts<<", n "<<n<<endl;
      GEODE_ASSERT(ss.standardize<8>().x==ss);
      GEODE_ASSERT(false);
    }
    // Does the board occur in the section?
    const board_t sboard = transform_board(symmetry_t(g,0),board);
    for (int i=0;i<4;i++) {
      RawArray<const quadrant_t> bucket = sorted.get(ss.counts[i]);
      const quadrant_t q = quadrant(sboard,i);
      GEODE_ASSERT(std::binary_search(bucket.begin(),bucket.end(),rotation_standardize_quadrant(q).x));
    }
  }
}

bool sorted_array_is_subset(RawArray<const board_t> boards0, RawArray<const board_t> boards1) {
  if (boards0.size()>boards1.size())
    return false;
  int i1 = 0;
  for (int i0=0;i0<boards0.size();i0++) {
    const board_t b0 = boards0[i0];
    for (;;) {
      if (b0==boards1[i1])
        break;
      if (b0<boards1[i1])
        return false;
      i1++;
      if (i1==boards1.size())
        return false;
    }
    i1++;
  }
  return true;
}

}
using namespace pentago;
using namespace geode::python;

void wrap_all_boards() {
  GEODE_FUNCTION(all_boards)
  GEODE_FUNCTION(distinguishing_hash_bits)
  GEODE_FUNCTION(minimal_quadrants)
  GEODE_FUNCTION(all_boards_sections)
  GEODE_FUNCTION(all_boards_stats)
  GEODE_FUNCTION(all_boards_list)
  GEODE_FUNCTION(all_boards_sample_test)
  GEODE_FUNCTION(all_boards_section_sizes)
  GEODE_FUNCTION(sorted_array_is_subset)
}
