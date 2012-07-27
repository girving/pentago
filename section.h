// Sections: sets of positions classified by stone counts in each quadrant
#pragma once

#include <pentago/board.h>
#include <other/core/array/RawArray.h>
#include <other/core/vector/Vector.h>
#include <other/core/structure/Tuple.h>
namespace pentago {

using std::ostream;

// Count stones in a quadrant
static inline Vector<uint8_t,2> count(quadrant_t q) {
  return vec((uint8_t)popcount(unpack(q,0)),(uint8_t)popcount(unpack(q,1)));
}

// All rotation minimal quadrants with the given numbers of stones, plus the number moved by reflections
Tuple<RawArray<const quadrant_t>,int> rotation_minimal_quadrants(int black, int white);
Tuple<RawArray<const quadrant_t>,int> rotation_minimal_quadrants(Vector<uint8_t,2> counts);

// Slices of rotation_minimal_quadrants
RawArray<const quadrant_t> safe_rmin_slice(Vector<uint8_t,2> counts, int lo, int hi);
RawArray<const quadrant_t> safe_rmin_slice(Vector<uint8_t,2> counts, Range<int> range);

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

  // Given s, find global symmetry g minimizing g(s) and return g(s),g
  template<int symmetries> Tuple<section_t,uint8_t> standardize() const;

  // Quadrant i is mapped to result[i].  Warnings: slower than necessary.
  static Vector<int,4> quadrant_permutation(uint8_t symmetry);
};

ostream& operator<<(ostream& output, section_t section);
PyObject* to_python(const section_t& section);
} namespace other {
template<> struct FromPython<pentago::section_t>{static pentago::section_t convert(PyObject* object);};
template<> struct is_packed_pod<pentago::section_t>:public mpl::true_{}; // Make section_t hashable
} namespace pentago {

// Find the rotation minimizing the quadrant value, and return minimized_quadrant, rotation
Tuple<quadrant_t,uint8_t> rotation_standardize_quadrant(quadrant_t q);

// Generate a random board within the given section.  Warning: fairly slow.
board_t random_board(Random& random, const section_t& section);

// Count stones in all quadrants
section_t count(board_t board);

}
