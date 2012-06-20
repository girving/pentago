// Multidimensional superscore filtering to precondition zlib compression
#pragma once

#include "superscore.h"
#include <other/core/utility/CopyConst.h>
namespace pentago {

void interleave(RawArray<Vector<super_t,2>> data);
void uninterleave(RawArray<Vector<super_t,2>> data);
Array<uint8_t> compact(Array<Vector<super_t,2>> src);
Array<Vector<super_t,2>> uncompact(Array<const uint8_t> src);

template<class TA> static RawArray<typename CopyConst<uint8_t,typename TA::Element>::type> char_view(const TA& data) {
  uint64_t size = sizeof(typename TA::Element)*(size_t)data.size();
  OTHER_ASSERT(size<(uint64_t)1<<31);
  typedef typename CopyConst<uint8_t,typename TA::Element>::type C;
  return RawArray<C>(size,(C*)data.data());
}

template<class TA> static Array<typename CopyConst<uint8_t,typename TA::Element>::type> char_view_own(const TA& data) {
  uint64_t size = sizeof(typename TA::Element)*(size_t)data.size();
  OTHER_ASSERT(size<(uint64_t)1<<31);
  typedef typename CopyConst<uint8_t,typename TA::Element>::type C;
  return Array<C>(size,(C*)data.data(),data.borrow_owner());
}

}
