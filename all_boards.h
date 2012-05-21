// Board enumeration
#pragma once

#include "board.h"
#include <other/core/array/NestedArray.h>
namespace pentago {

struct Section {
  Vector<Vector<uint8_t,2>,4> counts;

  Section() {}

  Section(const Vector<Vector<uint8_t,2>,4>& counts)
    : counts(counts) {}

  uint64_t sig() const {
    uint64_t s;
    memcpy(&s,&counts,8);
    return s;
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

  Section transform(uint8_t global) const;
  Tuple<Section,uint8_t> standardize() const;
};

class AllBoards : public Object {
public:
  OTHER_DECLARE_TYPE

  // Sorted lists of quadrants minimal w.r.t. rotations but *not* reflections.
  // The outer nested array dimension is indexed by tri(b,w), where b and w are the counts of black and white stones.
  // rmins partitioned by the number of black and white stones.
  const NestedArray<const quadrant_t> rmins;

  static const int buckets = 10*(10+1)/2; // rmins.size()

protected:
  AllBoards();
public:
  ~AllBoards();

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

  uint64_t size(Section s) const {
    return (uint64_t)rmins.size(tri(s.counts[0]))*rmins.size(tri(s.counts[1]))*rmins.size(tri(s.counts[2]))*rmins.size(tri(s.counts[3]));
  }

  // Enumerate the different ways n stones can be distributed into the four quadrants
  Array<Section> sections(int n, bool standardized) const;

  // Print statistics about the set of n stone positions, and return the total number including redundancies
  uint64_t stats(int n) const;

  // Enumerate all n stone positions
  Array<board_t> list(int n) const;

  // Test our enumeration
  void test(int n, int steps) const;

  // Given two sorted lists of boards, check that the first is contained in the second
  static bool is_subset(RawArray<const board_t> boards0, RawArray<const board_t> boards1);
};
}
