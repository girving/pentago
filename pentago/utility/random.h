// Class Random
#pragma once

#include "pentago/utility/uint128.h"
#include "pentago/utility/vector.h"
#include "pentago/utility/threefry.h"
namespace pentago {

using std::enable_if_t;
using std::is_floating_point_v;
using std::is_integral_v;
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

  template<class F> F uniform() {  // in [0,1)
    constexpr bool is_float = std::is_same<F, float>::value;
    static_assert(is_float || std::is_same<F, double>::value);
    return is_float ? F(0x1p-32) * bits<uint32_t>() : F(0x1p-64) * bits<uint64_t>();
  }

  template<class I> I uniform(const I n) {  // in [0,n)
    return uniform(I(0), n);
  }

  // Integer version for [a,b)
  template<class I> enable_if_t<is_integral_v<I>,I> uniform(const I a, const I b) {
    static_assert(sizeof(I) <= 8, "uint128_t values would require rejection sampling");
    typedef std::make_unsigned_t<I> UI;
    return a + I(bits<UI>() % UI(b - a));
  }
  
  // Floating point version for [a,b)
  template<class F> enable_if_t<is_floating_point_v<F>,F> uniform(const F a, const F b) {
    return a + uniform<F>() * (b - a);
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
