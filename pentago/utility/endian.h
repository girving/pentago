// Endianness utilities
#pragma once

#include <geode/array/view.h>
#include <boost/detail/endian.hpp>
namespace pentago {

using namespace geode;

// Note: All little endian conversion functions are their own inverse

#if defined(BOOST_LITTLE_ENDIAN)

// Our output formats are little endian, so no swapping necessary
template<class T> static inline const T& to_little_endian(const T& x) { return x; }
template<class TA> static inline void to_little_endian_inplace(const TA& data) {}

#elif defined(BOOST_BIG_ENDIAN)

// Handle all single byte types, and enforce exact matches for everything else
template<class T> static inline T to_little_endian(const T x) {
  BOOST_STATIC_ASSERT(sizeof(T)==1);
  return x;
}

static inline uint16_t to_little_endian(uint16_t x) {
  return x<<8|x>>8;
}

static inline uint32_t to_little_endian(uint32_t x) {
  const uint32_t lo = 0x00ff00ff;
  x = (x&lo)<<8|(x>>8&lo);
  x = x<<16|x>>16;
  return x;
}

static inline uint64_t to_little_endian(uint64_t x) {
  const uint64_t lo1 = 0x00ff00ff00ff00ff,
                 lo2 = 0x0000ffff0000ffff;
  x = (x&lo1)<<8|(x>>8&lo1);
  x = (x&lo2)<<16|(x>>16&lo2);
  x = x<<32|x>>32;
  return x;
}

static inline super_t to_little_endian(const super_t s) {
  return super_t(to_little_endian(s.d),
                 to_little_endian(s.c),
                 to_little_endian(s.b),
                 to_little_endian(s.a));
}

static inline section_t to_little_endian(const section_t s) {
  return s;
}

static inline supertensor_blob_t to_little_endian(supertensor_blob_t blob) {
  blob.uncompressed_size = to_little_endian(blob.uncompressed_size);
  blob.compressed_size = to_little_endian(blob.compressed_size);
  blob.offset = to_little_endian(blob.offset);
  return blob;
}

template<class T,int d> static inline Vector<T,d> to_little_endian(const Vector<T,d>& x) {
  Vector<T,d> le;
  for (int i=0;i<d;i++)
    le[i] = to_little_endian(x[i]);
  return le;
}

template<class TA> static inline void to_little_endian_inplace(const TA& data) {
  for (auto& x : data)
    x = to_little_endian(x);
}

#endif

// Same as to_little_endian, but useful for documentation purposes
template<class T> static inline T from_little_endian(const T& x) {
  return to_little_endian(x);
}

}
