// Endian utilities
#pragma once

#include <boost/endian/conversion.hpp>
namespace pentago {

template<class A> void to_little_endian_inplace(const A& data) {
#ifdef BOOST_BIG_ENDIAN
  for (auto& x : flat) boost::endian::native_to_little_inplace(x); 
#endif
}

}
