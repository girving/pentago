// ceil_div(a,b) = ceil(a/b), but with integers
#pragma once

#include <boost/type_traits/is_integral.hpp>
#include <boost/mpl/assert.hpp>
#include <cassert>
namespace pentago {

template<class TV,class T> static inline TV ceil_div(TV a, T b) {
  BOOST_MPL_ASSERT((boost::is_integral<T>));
  assert(b > 0);
  return (a+b-1)/b;
}

}
