// Sections: sets of positions classified by stone counts in each quadrant
//
// A section consists of all boards with the same numbers of black and white
// stones in each quadrant.  For example, the largest section is 33333333,
// and consists of stones with 3 stones of each color in each quadrant.
// Sections play a key role in the distribution of data in the endgame solver,
// and in the resulting output data formats.
#pragma once

#include "pentago/base/board.h"
#include "pentago/utility/array.h"
#include "pentago/utility/range.h"
#include "pentago/utility/vector.h"
namespace pentago {

using std::ostream;
using std::tuple;

// Count stones in a quadrant
static inline Vector<uint8_t,2> count(quadrant_t q) {
  return vec((uint8_t)popcount(unpack(q,0)),(uint8_t)popcount(unpack(q,1)));
}

// All rotation minimal quadrants with the given numbers of stones, plus the number moved by reflections
tuple<RawArray<const quadrant_t>,int> rotation_minimal_quadrants(int black, int white);
tuple<RawArray<const quadrant_t>,int> rotation_minimal_quadrants(Vector<uint8_t,2> counts);

// Slices of rotation_minimal_quadrants
RawArray<const quadrant_t> safe_rmin_slice(Vector<uint8_t,2> counts, int lo, int hi);
RawArray<const quadrant_t> safe_rmin_slice(Vector<uint8_t,2> counts, Range<int> range);

struct section_t {
  Vector<Vector<uint8_t,2>,4> counts;

  section_t() {}

  section_t(const Vector<Vector<uint8_t,2>,4>& counts)
    : counts(counts) {}

  uint64_t sig() const {
    // This used to be a memcpy, but is now explicitly little endian for portability.
    // Ideally this would be merged with microsig below, but unfortunately it shows up in file formats.
    return            counts[0][0]     +((uint64_t)counts[0][1]<<8)
          +((uint64_t)counts[1][0]<<16)+((uint64_t)counts[1][1]<<24)
          +((uint64_t)counts[2][0]<<32)+((uint64_t)counts[2][1]<<40)
          +((uint64_t)counts[3][0]<<48)+((uint64_t)counts[3][1]<<56);
  }

  uint32_t microsig() const {
    return       counts[0][0]     +((int)counts[0][1]<<4)
          +((int)counts[1][0]<<8) +((int)counts[1][1]<<12)
          +((int)counts[2][0]<<16)+((int)counts[2][1]<<20)
          +((int)counts[3][0]<<24)+((int)counts[3][1]<<28);
  }

  bool operator==(const section_t& s) const {
    return counts==s.counts;
  }

  bool operator!=(const section_t& s) const {
    return counts!=s.counts;
  }

  bool operator<(const section_t& s) const {
    return sig()<s.sig();
  }

  bool operator>(const section_t& s) const {
    return sig()>s.sig();
  }

  // Number of rotational minimal quadrants along each dimension
  Vector<int,4> shape() const;

  // Total number of positions in section
  uint64_t size() const;

  int sum() const {
    return counts.sum().sum();
  }

  Vector<int,4> sums() const {
    return Vector<int,4>(counts[0].sum(),counts[1].sum(),counts[2].sum(),counts[3].sum());
  }

  bool valid() const;
  section_t transform(uint8_t global) const;
  section_t child(int quadrant) const;
  section_t parent(int quadrant) const;

  // Given s, find global symmetry g minimizing g(s) and return g(s),g
  template<int symmetries> tuple<section_t,uint8_t> standardize() const;

  // Quadrant i is mapped to result[i].  Warnings: slower than necessary.
  static Vector<int,4> quadrant_permutation(uint8_t symmetry);
};

ostream& operator<<(ostream& output, section_t section);

// Find the rotation minimizing the quadrant value, and return minimized_quadrant, rotation
tuple<quadrant_t,uint8_t> rotation_standardize_quadrant(quadrant_t q);

// Generate a random board within the given section.  Warning: fairly slow.
board_t random_board(Random& random, const section_t& section);

// Count stones in all quadrants
section_t count(board_t board);

// Show counts and rmin indices
string show_board_rmins(const board_t board);

// Endian conversion is trivial for section_t
static inline section_t endian_reverse(const section_t s) {
  return s;
}

static inline size_t hash_value(const section_t& s) {
  size_t h = 0;
  for (const auto& x : s.counts)
    for (const auto& y : x)
      hash_combine(h, y);
  return h;
}

}  // namespace pentago
namespace std {
template<> struct hash<pentago::section_t> {
  size_t operator()(const pentago::section_t& s) const {
    return hash_value(s);
  }
};
}  // namespace std
