#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/char_view.h"
#include "pentago/utility/endian.h"
namespace pentago {

string sha1(RawArray<const uint8_t> data);

template<class A> string portable_hash(const A& data) {
#ifdef PENTAGO_BIG_ENDIAN
  const auto flat = asarray(data).flat().copy();
  for (auto& x : flat) boost::endian::native_to_little_inplace(x);
#else
  const auto flat = asarray(data).flat();
#endif
  return sha1(char_view(flat).const_());
}

}
