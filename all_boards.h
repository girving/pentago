// Board enumeration
#pragma once

#include "board.h"
#include <other/core/array/NestedArray.h>
namespace pentago {

// Count stones in a quadrant
static inline Vector<uint8_t,2> count(quadrant_t q) {
  return vec((uint8_t)popcount(unpack(q,0)),(uint8_t)popcount(unpack(q,1)));
}

// All rotation minimal quadrants with the given numbers of stones
static inline RawArray<const quadrant_t> rotation_minimal_quadrants(int black, int white) {
  assert(0<=black && 0<=white && black+white<=9);
  const int i = ((black*(21-black))>>1)+white;
  const uint16_t lo = rotation_minimal_quadrants_offsets[i],
                 hi = rotation_minimal_quadrants_offsets[i+1];
  return RawArray<const quadrant_t>(hi-lo,rotation_minimal_quadrants_flat+lo);
}
static inline RawArray<const quadrant_t> rotation_minimal_quadrants(Vector<uint8_t,2> counts) {
  return rotation_minimal_quadrants(counts.x,counts.y);
}

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

  uint64_t size() const {
    return (uint64_t)rotation_minimal_quadrants(counts[0]).size()
                    *rotation_minimal_quadrants(counts[1]).size()
                    *rotation_minimal_quadrants(counts[2]).size()
                    *rotation_minimal_quadrants(counts[3]).size();
  }

  Section transform(uint8_t global) const;
  Tuple<Section,uint8_t> standardize() const;
};

// Enumerate the different ways n stones can be distributed into the four quadrants
Array<Section> all_boards_sections(int n, bool standardized=true);

// Print statistics about the set of n stone positions, and return the total number including redundancies
uint64_t all_boards_stats(int n);

// Enumerate all supersymmetric n stone positions, with some redundancy
Array<board_t> all_boards_list(int n);

// Test our enumeration
void all_boards_sample_test(int n, int steps);

// Given two sorted lists of boards, check that the first is contained in the second
bool sorted_array_is_subset(RawArray<const board_t> boards0, RawArray<const board_t> boards1);

}
