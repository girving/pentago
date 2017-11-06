// Boxes
#pragma once

#include "pentago/utility/vector.h"
namespace pentago {

template<class TV> class Box {
  typedef scalar_type<TV> T;
public:
  TV min, max;

  Box() : min(), max() {}
  Box(const TV& x) : min(x), max(x) {}
  Box(const TV& min, const TV& max) : min(min), max(max) {}
  template<class SV> explicit Box(const Box<SV>& box) : min(box.min), max(box.max) {}

  TV shape() const { return max - min; }
  TV center() const { return T(0.5)*(min + max); }
  TV clamp(const TV x) const { return pentago::clamp(x, min, max); }
  auto volume() const { return shape().product(); } 

  Box& operator+=(const Box<TV> box) { min += box.min; max += box.max; return *this; }

  void enlarge(const TV x) {
    min = cwise_min(min, x);
    max = cwise_max(max, x);
  }

  void enlarge(const Box<TV> box) {
    min = cwise_min(min, box.min);
    max = cwise_max(max, box.max);
  }

  bool contains(const TV x) const {
    // TODO: Optimize
    return x == clamp(x);
  }

  Box thickened(const T x) const {
    return Box(min - x, max + x);
  }
};

template<class TV> static inline Box<TV> intersect(const Box<TV>& x, const Box<TV>& y) {
  return Box<TV>(cwise_max(x.min, y.min), cwise_min(x.max, y.max));
}

}
