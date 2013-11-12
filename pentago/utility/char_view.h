// Cast a typed array down to bytes
#pragma once

#include <geode/array/RawArray.h>
#include <geode/utility/CopyConst.h>
namespace pentago {

template<class TA> static RawArray<typename CopyConst<uint8_t,typename TA::Element>::type> char_view(const TA& data) {
  uint64_t size = sizeof(typename TA::Element)*(size_t)data.size();
  GEODE_ASSERT(size<(uint64_t)1<<31);
  typedef typename CopyConst<uint8_t,typename TA::Element>::type C;
  return RawArray<C>(size,(C*)data.data());
}

template<class TA> static Array<typename CopyConst<uint8_t,typename TA::Element>::type> char_view_own(const TA& data) {
  uint64_t size = sizeof(typename TA::Element)*(size_t)data.size();
  GEODE_ASSERT(size<(uint64_t)1<<31);
  typedef typename CopyConst<uint8_t,typename TA::Element>::type C;
  return Array<C>(size,(C*)data.data(),data.borrow_owner());
}

}
