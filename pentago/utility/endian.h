// Endian utilities
#pragma once

#ifndef __wasm__
#include <boost/endian/conversion.hpp>
#endif
namespace pentago {

#ifdef __wasm__

#define PENTAGO_LITTLE_ENDIAN  // Emscripten is little endian

#else

#if defined(BOOST_BIG_ENDIAN)
#define PENTAGO_BIG_ENDIAN
#elif defined(BOOST_LITTLE_ENDIAN)
#define PENTAGO_LITTLE_ENDIAN
#else
#error "Unknown endian"
#endif

template<class A> void to_little_endian_inplace(const A& data) {
#ifdef PENTAGO_BIG_ENDIAN
  for (auto& x : flat) boost::endian::native_to_little_inplace(x); 
#endif
}

#endif  // __wasm__

}
