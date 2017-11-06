// Class Random
#pragma once

#include "pentago/utility/uint128.h"
#include "pentago/utility/vector.h"
#include "pentago/utility/threefry.h"
namespace pentago {

using std::swap;

class Random {
public:
  // Counter mode, so we get to expose seed and counter as mutable fields!
  uint128_t seed;
  uint128_t counter;

  explicit Random(uint128_t seed) : seed(seed), counter(0) {}

  // Disallow implicit copy since copies must be done with care
  Random(const Random&) = delete;
  void operator=(const Random&) = delete;

  template<class I> I bits() {
    static_assert(std::is_integral<I>::value);
    return static_cast<I>(threefry(seed, counter++));
  }

  template<class I> I uniform(const I n) {  // in [0,n)
    return uniform(I(0), n);
  }

  template<class I> I uniform(const I a, const I b) {  // in [a,b)
    static_assert(std::is_integral<I>::value);
    static_assert(sizeof(I) <= 8, "uint128_t values would require rejection sampling");
    typedef std::make_unsigned_t<I> UI;
    return a + I(bits<UI>() % UI(b - a));
  }

  template<class I,int d> Vector<I,d> uniform(const Vector<I,d>& min, const Vector<I,d>& max) {
    Vector<I,d> r;
    for (int i = 0; i < d; i++) r[i] = uniform<I>(min[i], max[i]);
    return r;
  }

  template<class TA> void shuffle(TA& v) {
    int n = v.size();
    for (int i=0;i<n-1;i++) {
      int j = uniform<int>(i,n);
      if (i!=j) swap(v[i], v[j]);
    }
  }
};

}
