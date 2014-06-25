// Endianness utilities
//
// Conversion from native endianness to/from little endianness, useful
// for writing platform independent files on Blue Gene.
#pragma once

#include <geode/array/view.h>
#include <geode/utility/endian.h>
namespace pentago {

using namespace geode;
using geode::flip_endian;

#if GEODE_ENDIAN == GEODE_BIG_ENDIAN
static inline super_t flip_endian(const super_t s) {
  return super_t(flip_endian(s.d),
                 flip_endian(s.c),
                 flip_endian(s.b),
                 flip_endian(s.a));
}
#endif

static inline section_t flip_endian(const section_t s) {
  return s;
}

static inline supertensor_blob_t flip_endian(supertensor_blob_t blob) {
  blob.uncompressed_size = flip_endian(blob.uncompressed_size);
  blob.compressed_size = flip_endian(blob.compressed_size);
  blob.offset = flip_endian(blob.offset);
  return blob;
}

template<class TA> static inline void to_little_endian_inplace(const TA& data) {
  for (auto& x : data)
    x = to_little_endian(x);
}

}
