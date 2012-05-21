// Board enumeration

#include "all_boards.h"
#include "symmetry.h"
#include "table.h"
#include "count.h"
#include <other/core/array/NdArray.h>
#include <other/core/array/sort.h>
#include <other/core/math/popcount.h>
#include <other/core/math/uint128.h>
#include <other/core/python/Class.h>
#include <other/core/python/module.h>
#include <other/core/random/Random.h>
#include <other/core/structure/Hashtable.h>
#include <other/core/utility/interrupts.h>
#include <other/core/utility/const_cast.h>
#include <other/core/vector/Interval.h>
namespace pentago {

using namespace other;
using std::cout;
using std::endl;
using std::vector;

OTHER_DEFINE_TYPE(AllBoards)

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

// Quadrants minimal w.r.t. rotations and reflections
static Array<quadrant_t> minimal_quadrants() {
  Array<quadrant_t> mins;
  for (quadrant_t q=0;q<quadrant_count;q++)
    if (superstandardize(q).x==q)
      mins.append(q);
  return mins;
}

static quadrant_t rotation_standardize_quadrant(quadrant_t q) {
  quadrant_t minq = q;
  side_t s[4][2];
  s[0][0] = unpack(q,0);
  s[0][1] = unpack(q,1);
  for (int r=0;r<3;r++) {
    for (int i=0;i<2;i++)
      s[r+1][i] = rotations[s[r][i]][0];
    minq = min(minq,pack(s[r+1][0],s[r+1][1]));
  }
  return minq;
}

Section Section::transform(uint8_t global) const {
  const int r = global&3;
  static uint8_t source[4][4] = {{0,1,2,3},{1,3,0,2},{3,2,1,0},{2,0,3,1}};
  Section t(vec(counts[source[r][0]],counts[source[r][1]],counts[source[r][2]],counts[source[r][3]]));
  if (global&4)
    swap(t.counts[0],t.counts[3]);
  return t;
}

Tuple<Section,uint8_t> Section::standardize() const {
  Section best = *this;
  uint8_t best_g = 0;
  for (int g=1;g<8;g++) {
    Section t = transform(g);
    if (best > t) {
      best = t;
      best_g = g;
    }
  }
  return tuple(best,best_g);
}

AllBoards::AllBoards() {
  // Sort quadrants minimal w.r.t. rotation only by the superstandardized value
  Array<quadrant_t> all_rmins;
  for (quadrant_t q=0;q<quadrant_count;q++)
    if (rotation_standardize_quadrant(q)==q)
      all_rmins.append(q);

  // Partition all_rmins based on the number of black and white stones
  Array<int> counts(buckets);
  for (auto q : all_rmins)
    counts[tri(count(q))]++;
  NestedArray<quadrant_t> rmins(counts,false);
  counts.zero();
  for (auto q : all_rmins) {
    int i = tri(count(q));
    rmins(i,counts[i]++) = q;
  }
  const_cast_(this->rmins) = rmins;
}

AllBoards::~AllBoards() {}

Array<Section> AllBoards::sections(int n, bool standardized) const {
  OTHER_ASSERT(0<=n && n<=36);
  const int white = n/2, black = n-white;
  Array<Section> sections;

  // Loop over possible counts in quadrant 3
  for (int b3=0;b3<=min(black,9);b3++) {
    for (int w3=0;w3<=min(white,9-b3);w3++) {
      const Vector<int,2> left3(black-b3,white-w3);
      // Loop over possible counts in quadrant 2
      const Interval<int> range_b2(0,min(left3.x,9));
      for (int b2=range_b2.min;b2<=range_b2.max;b2++) {
        const Interval<int> range_w2(max(0,left3.y-(18-(left3.x-b2))),min(left3.y,9-b2));
        for (int w2=range_w2.min;w2<=range_w2.max;w2++) {
          const Vector<int,2> left2 = left3-vec(b2,w2);
          // Loop over possible counts in quadrant 1
          const Interval<int> range_b1(max(0,left2.x-9),min(left2.x,9));
          for (int b1=range_b1.min;b1<=range_b1.max;b1++) {
            const Interval<int> range_w1(max(0,left2.y-(9-left2.x+b1)),min(left2.y,9-b1));
            for (int w1=range_w1.min;w1<=range_w1.max;w1++) {
              const Vector<int,2> left1 = left2-vec(b1,w1);
              // Quadrant 0's counts are now uniquely determined
              const int b0 = left1.x, w0 = left1.y;
              // We've found a section!
              Section s(vec(Vector<uint8_t,2>(b0,w0),Vector<uint8_t,2>(b1,w1),Vector<uint8_t,2>(b2,w2),Vector<uint8_t,2>(b3,w3)));
              if (!standardized || s.standardize().x==s)
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

uint64_t AllBoards::stats(int n) const {
  Array<Section> sections = this->sections(n,false);
  int reduced_sections = 0;
  uint64_t max_section = 0;
  uint64_t total = 0;
  uint64_t reduced_total = 0;
  for (Section s : sections) {
    const uint64_t size = this->size(s);
    max_section = max(max_section,size);
    total += size;
    if (s.standardize().x==s) {
      reduced_sections++;
      reduced_total += size;
    }
  }
  const uint64_t exact = supercount_boards(n);
  cout <<"n = "<<n<<", simple count = "<<reduced_total<<", ratio = "<<(double)reduced_total/exact<<", unreduced ratio = "<<(double)total/exact
       <<", reduced sections = "<<reduced_sections<<", unreduced sections = "<<sections.size()<<", max section = "<<max_section<<", average section = "<<(double)total/sections.size()<<endl;
  OTHER_ASSERT(8*reduced_sections>=sections.size());
  return reduced_total;
}

Array<board_t> AllBoards::list(int n) const {
  Array<Section> sections = this->sections(n,true);

  // Make sure we fit into Array
  uint64_t large_count = 0;
  for (Section s : sections)
    large_count += size(s);
  const int small_count = large_count;
  OTHER_ASSERT(small_count>0 && small_count==large_count);

  // Collect boards
  Array<board_t> list;
  list.preallocate(small_count);
  for (Section s : sections) {
    check_interrupts();
    RawArray<const quadrant_t> bucket0 = rmins[tri(s.counts[0])],
                               bucket1 = rmins[tri(s.counts[1])],
                               bucket2 = rmins[tri(s.counts[2])],
                               bucket3 = rmins[tri(s.counts[3])];
    for (auto q0 : bucket0)
      for (auto q1 : bucket1)
        for (auto q2 : bucket2)
          for (auto q3 : bucket3)
            list.append_assuming_enough_space(quadrants(q0,q1,q2,q3));
  }
  return list;
}

void AllBoards::test(int n, int steps) const {
  Array<Section> sections = this->sections(n,true);

  // Generate a bunch of random boards, and check that each one occurs in a section
  Ref<Random> random = new_<Random>(175131);
  for (int step=0;step<steps;step++) {
    const board_t board = random_board(random,n);
    const Section s(vec(count(quadrant(board,0)),count(quadrant(board,1)),count(quadrant(board,2)),count(quadrant(board,3))));
    Section ss;uint8_t g;s.standardize().get(ss,g);
    // Does this section exist?
    int si = std::lower_bound(sections.begin(),sections.end(),ss)-sections.begin();
    if (!(sections.valid(si) && sections[si]==ss)) {
      cout << "missing section "<<s.counts<<", standard "<<ss.counts<<", n "<<n<<endl;
      OTHER_ASSERT(ss.standardize().x==ss);
      OTHER_ASSERT(false);
    }
    // Does the board occur in the section?
    const board_t sboard = transform_board(symmetry_t(g,0),board);
    for (int i=0;i<4;i++) {
      RawArray<const quadrant_t> bucket = rmins[tri(ss.counts[i])]; 
      const quadrant_t q = quadrant(sboard,i);
      OTHER_ASSERT(std::binary_search(bucket.begin(),bucket.end(),rotation_standardize_quadrant(q)));
    }
  }
}

bool AllBoards::is_subset(RawArray<const board_t> boards0, RawArray<const board_t> boards1) {
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
using namespace other::python;

void wrap_all_boards() {
  OTHER_FUNCTION(all_boards)
  OTHER_FUNCTION(distinguishing_hash_bits)
  OTHER_FUNCTION(minimal_quadrants)

  typedef AllBoards Self;
  Class<AllBoards>("AllBoards")
    .OTHER_INIT()
    .OTHER_METHOD(stats)
    .OTHER_METHOD(list)
    .OTHER_METHOD(test)
    .OTHER_METHOD(is_subset)
    ;
}
