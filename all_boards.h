// Board enumeration
#pragma once

#include "board.h"
#include <other/core/array/NestedArray.h>
namespace pentago {

using std::ostream;

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

struct section_t {
  Vector<Vector<uint8_t,2>,4> counts;

  section_t() {}

  section_t(const Vector<Vector<uint8_t,2>,4>& counts)
    : counts(counts) {}

  uint64_t sig() const {
    uint64_t s;
    memcpy(&s,&counts,8);
    return s;
  }

  bool operator==(const section_t& s) const {
    return counts==s.counts;
  }

  bool operator<(const section_t& s) const {
    return sig()<s.sig();
  }

  bool operator>(const section_t& s) const {
    return sig()>s.sig();
  }

  Vector<int,4> shape() const {
    return vec(rotation_minimal_quadrants(counts[0]).size(),
               rotation_minimal_quadrants(counts[1]).size(),
               rotation_minimal_quadrants(counts[2]).size(),
               rotation_minimal_quadrants(counts[3]).size());
  }

  uint64_t size() const {
    const auto s = shape();
    return (uint64_t)s[0]*s[1]*s[2]*s[3];
  }

  int sum() const {
    return counts.sum().sum();
  }

  bool valid() const;
  section_t transform(uint8_t global) const;
  Tuple<section_t,uint8_t> standardize() const;
};

ostream& operator<<(ostream& output, section_t section);
PyObject* to_python(const section_t& section);
} namespace other {
template<> struct FromPython<pentago::section_t>{static pentago::section_t convert(PyObject* object);};
} namespace pentago {

// Enumerate the different ways n stones can be distributed into the four quadrants
Array<section_t> all_boards_sections(int n, bool standardized=true);

// Print statistics about the set of n stone positions, and return the total number including redundancies
uint64_t all_boards_stats(int n);

// Enumerate all supersymmetric n stone positions, with some redundancy
Array<board_t> all_boards_list(int n);

// Test our enumeration
void all_boards_sample_test(int n, int steps);

// Given two sorted lists of boards, check that the first is contained in the second
bool sorted_array_is_subset(RawArray<const board_t> boards0, RawArray<const board_t> boards1);

}
