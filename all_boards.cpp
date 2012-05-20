// Board enumeration

#include "board.h"
#include "symmetry.h"
#include "table.h"
#include "count.h"
#include <other/core/array/NdArray.h>
#include <other/core/array/NestedArray.h>
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

namespace {

class ApproximateBoards : public Object {
public:
  OTHER_DECLARE_TYPE

  // Quadrants minimal w.r.t. rotations and reflections
  const Array<const quadrant_t> mins;

  // Quadrants minimal w.r.t. rotations but *not* reflections, sorted in order of their superstandardized values.
  // The outer nested array dimension is indexed by tri(b,w), where b and w are the counts of black and white stones.
  // rmins partitioned by the number of black and white stones.  Indexed by tri(b,w)
  const NestedArray<const quadrant_t> rmins;

  // rmins in normal (nonsuperstandardized) order
  const NestedArray<const quadrant_t> sorted_rmins;

  static const int buckets = 10*(10+1)/2; // rmin_buckets.size()

  // Given a minimal quadrant q and stone counts (b,w), starts(q,tri(b,w)) is the least i s.t. index of superstandardize(rmin_buckets(tri(b,w),i)).x >= q
  const Array<const Vector<uint16_t,buckets>> starts;

protected:
  ApproximateBoards();
public:

  // Index into rmins
  static int tri(int b, int w) {
    assert(0<=b && 0<=w && b+w<=9);
    return ((b*(21-b))>>1)+w;
  }

  static int tri(Vector<uint8_t,2> bw) {
    return tri(bw.x,bw.y);
  }

  // Count stones in a quadrant
  static Vector<uint8_t,2> count(quadrant_t q) {
    return vec((uint8_t)popcount(unpack(q,0)),(uint8_t)popcount(unpack(q,1)));
  }

  // Compute statistics about all n stone positions
  uint64_t stats(int n) const;

  // List all boards with the given number of stones.  There will be a small number of duplicate nonsuperstandardized boards.
  Array<board_t> list(int n) const;

  struct Section {
    Vector<Vector<uint8_t,2>,4> counts;

    Section() {}

    Section(const Vector<Vector<uint8_t,2>,4>& counts)
      : counts(counts) {}

    Section transform(uint8_t global) const {
      const int r = global&3;
      static uint8_t source[4][4] = {{0,1,2,3},{1,3,0,2},{3,2,1,0},{2,0,3,1}};
      Section t(vec(counts[source[r][0]],counts[source[r][1]],counts[source[r][2]],counts[source[r][3]]));
      if (global&4)
        swap(t.counts[0],t.counts[3]);
      return t;
    }

    uint64_t sig() const {
      uint64_t s;
      memcpy(&s,&counts,8);
      return s;
    }

    Tuple<Section,uint8_t> standardize() const {
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

    bool operator==(const Section& s) const {
      return counts==s.counts;
    }

    bool operator<(const Section& s) const {
      return sig()<s.sig();
    }

    bool operator>(const Section& s) const {
      return sig()>s.sig();
    }
  };

  uint64_t size(Section s) const {
    return (uint64_t)rmins.size(tri(s.counts[0]))*rmins.size(tri(s.counts[1]))*rmins.size(tri(s.counts[2]))*rmins.size(tri(s.counts[3]));
  }

  // Simplified versions that don't bother with global symmetry standardization
  Array<Section> simple_sections(int n, bool standardized) const;
  uint64_t simple_stats(int n) const;
  Array<board_t> simple_list(int n) const;

  // Test simple enumeration
  void simple_test(int n, int steps) const;

  // Given two sorted lists of boards, check that the first is contained in the second
  bool is_subset(RawArray<const board_t> boards0, RawArray<const board_t> boards1) const;
};

OTHER_DEFINE_TYPE(ApproximateBoards)

ApproximateBoards::ApproximateBoards()
  : mins(minimal_quadrants()) {

  // Sort quadrants minimal w.r.t. rotation only by the superstandardized value
  Array<Vector<quadrant_t,2>> all_rmins;
  for (quadrant_t q=0;q<quadrant_count;q++)
    if (rotation_standardize_quadrant(q)==q)
      all_rmins.append(vec(q,(quadrant_t)superstandardize(q).x));
  sort(all_rmins,field_comparison(&Vector<quadrant_t,2>::y));

  // Partition rmins based on the number of black and white stones
  Array<int> counts(buckets);
  for (auto q : all_rmins)
    counts[tri(count(q.x))]++;
  NestedArray<quadrant_t> partition(counts,false),
                          standard_partition = NestedArray<quadrant_t>::empty_like(partition);
  counts.zero();
  for (auto q : all_rmins) {
    int i = tri(count(q.x));
    partition(i,counts[i]) = q.x;
    standard_partition(i,counts[i]) = q.x;
    counts[i]++;
  }
  const_cast_(rmins) = partition;

  const_cast_(sorted_rmins) = rmins.copy();
  for (int b=0;b<buckets;b++)
    sort(sorted_rmins[b].const_cast_());

  // Compute starts
  Array<Vector<uint16_t,buckets>> all_starts(mins.size(),false);
  for (int i=0;i<mins.size();i++) {
    const quadrant_t q = mins[i];
    Vector<uint16_t,buckets>& starts = all_starts[i];
    for (int bw=0;bw<buckets;bw++) {
      RawArray<const quadrant_t> bucket = standard_partition[bw];
      starts[bw] = std::lower_bound(bucket.begin(),bucket.end(),q)-bucket.begin();
    }
  }
  const_cast_(starts) = all_starts;
}

uint64_t ApproximateBoards::stats(int n) const {
  OTHER_ASSERT(0<=n && n<=36);
  const int white = n/2, black = n-white;

  // Interesting statistics
  int sections = 0;
  uint64_t max_section = 0;
  uint64_t total = 0;

  // Quadrant 3 is always minimal
  for (int i3=0;i3<mins.size();i3++) {
    const quadrant_t q3 = mins[i3];
    const Vector<int,2> left3 = vec(black,white)-Vector<int,2>(count(q3));
    const Vector<uint16_t,buckets>& starts = this->starts[i3];
    // Loop over possible counts in quadrant 2
    const Interval<int> range_b2(0,min(left3.x,9));
    for (int b2=range_b2.min;b2<=range_b2.max;b2++) {
      const Interval<int> range_w2(max(0,left3.y-(18-(left3.x-b2))),min(left3.y,9-b2));
      for (int w2=range_w2.min;w2<=range_w2.max;w2++) {
        const int bw2 = tri(b2,w2);
        RawArray<const quadrant_t> bucket2 = rmins[bw2].slice(starts[bw2],rmins.size(bw2));
        const Vector<int,2> left2 = left3-vec(b2,w2);
        // Loop over possible counts in quadrant 1
        const Interval<int> range_b1(max(0,left2.x-9),min(left2.x,9));
        for (int b1=range_b1.min;b1<=range_b1.max;b1++) {
          const Interval<int> range_w1(max(0,left2.y-(9-left2.x+b1)),min(left2.y,9-b1));
          for (int w1=range_w1.min;w1<=range_w1.max;w1++) {
            const int bw1 = tri(b1,w1);
            RawArray<const quadrant_t> bucket1 = rmins[bw1].slice(starts[bw1],rmins.size(bw1));
            const Vector<int,2> left1 = left2-vec(b1,w1);
            // Quadrant 0's counts are now uniquely determined
            const int b0 = left1.x, w0 = left1.y;
            const int bw0 = tri(b0,w0);
            RawArray<const quadrant_t> bucket0 = rmins[bw0].slice(starts[bw0],rmins.size(bw0));
            // Compute statistics
            sections++;
            const uint64_t size = bucket0.size()*bucket1.size()*bucket2.size();
            max_section = max(max_section,size);
            total += size;
          }
        }
      }
    }
  }
  uint64_t exact = supercount_boards(n);
  cout << "n = "<<n<<", count = "<<total<<", ratio = "<<(double)total/exact<<", sections = "<<sections<<", max section = "<<max_section<<", average section = "<<(double)total/sections<<endl;
  return total;
}

Array<ApproximateBoards::Section> ApproximateBoards::simple_sections(int n, bool standardized) const {
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

uint64_t ApproximateBoards::simple_stats(int n) const {
  Array<Section> sections = simple_sections(n,false);
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

Array<board_t> ApproximateBoards::simple_list(int n) const {
  Array<Section> sections = simple_sections(n,true);

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

void ApproximateBoards::simple_test(int n, int steps) const {
  Array<Section> sections = simple_sections(n,true);

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
      RawArray<const quadrant_t> bucket = sorted_rmins[tri(ss.counts[i])]; 
      const quadrant_t q = quadrant(sboard,i);
      OTHER_ASSERT(std::binary_search(bucket.begin(),bucket.end(),rotation_standardize_quadrant(q)));
    }
  }
}

// Generate mostly superstandardized boards explicitly without (much) duplication
Array<board_t> ApproximateBoards::list(int n) const {
  // Make sure we fit into Array
  OTHER_ASSERT(0<=n && n<=36);
  const uint128_t large_count = 2*supercount_boards(n);
  const int small_count = large_count;
  OTHER_ASSERT(small_count>0 && small_count==large_count);
  const int white = n/2, black = n-white;
  Array<board_t> list;

  // Quadrant 3 is always minimal
  for (int i3=0;i3<mins.size();i3++) {
    const quadrant_t q3 = mins[i3];
    const Vector<int,2> left3 = vec(black,white)-Vector<int,2>(count(q3));
    const Vector<uint16_t,buckets>& starts = this->starts[i3];
    // Loop over possible counts in quadrant 2
    const Interval<int> range_b2(0,min(left3.x,9));
    for (int b2=range_b2.min;b2<=range_b2.max;b2++) {
      const Interval<int> range_w2(max(0,left3.y-(18-(left3.x-b2))),min(left3.y,9-b2));
      for (int w2=range_w2.min;w2<=range_w2.max;w2++) {
        const int bw2 = tri(b2,w2);
        RawArray<const quadrant_t> bucket2 = rmins[bw2].slice(starts[bw2],rmins.size(bw2));
        const Vector<int,2> left2 = left3-vec(b2,w2);
        // Loop over possible counts in quadrant 1
        const Interval<int> range_b1(max(0,left2.x-9),min(left2.x,9));
        for (int b1=range_b1.min;b1<=range_b1.max;b1++) {
          const Interval<int> range_w1(max(0,left2.y-(9-left2.x+b1)),min(left2.y,9-b1));
          for (int w1=range_w1.min;w1<=range_w1.max;w1++) {
            const int bw1 = tri(b1,w1);
            RawArray<const quadrant_t> bucket1 = rmins[bw1].slice(starts[bw1],rmins.size(bw1));
            const Vector<int,2> left1 = left2-vec(b1,w1);
            // Quadrant 0's counts are now uniquely determined
            const int b0 = left1.x, w0 = left1.y;
            const int bw0 = tri(b0,w0);
            RawArray<const quadrant_t> bucket0 = rmins[bw0].slice(starts[bw0],rmins.size(bw0));
            // Loop over all triples from the three buckets
            for (quadrant_t q0 : bucket0)
              for (quadrant_t q1 : bucket1)
                for (quadrant_t q2 : bucket2)
                  list.append(quadrants(q0,q1,q2,q3));
          }
        }
      }
    }
  }
  return list;
}

bool ApproximateBoards::is_subset(RawArray<const board_t> boards0, RawArray<const board_t> boards1) const {
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

}
using namespace pentago;
using namespace other::python;

void wrap_all_boards() {
  OTHER_FUNCTION(all_boards)
  OTHER_FUNCTION(distinguishing_hash_bits)
  OTHER_FUNCTION(minimal_quadrants)

  typedef ApproximateBoards Self;
  Class<ApproximateBoards>("ApproximateBoards")
    .OTHER_INIT()
    .OTHER_METHOD(list)
    .OTHER_METHOD(stats)
    .OTHER_METHOD(simple_stats)
    .OTHER_METHOD(simple_list)
    .OTHER_METHOD(simple_test)
    .OTHER_METHOD(is_subset)
    ;
}
