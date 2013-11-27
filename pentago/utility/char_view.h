// Cast a typed array down to bytes
#pragma once

#include <pentago/utility/debug.h>
#include <geode/array/RawArray.h>
#include <geode/utility/CopyConst.h>
namespace pentago {

template<class TA> static RawArray<typename CopyConst<uint8_t,typename TA::Element>::type> char_view(const TA& data) {
  const int size = CHECK_CAST_INT(sizeof(typename TA::Element)*(size_t)data.size());
  typedef typename CopyConst<uint8_t,typename TA::Element>::type C;
  return RawArray<C>(size,(C*)data.data());
}

template<class TA> static Array<typename CopyConst<uint8_t,typename TA::Element>::type> char_view_own(const TA& data) {
  const int size = CHECK_CAST_INT(sizeof(typename TA::Element)*(size_t)data.size());
  typedef typename CopyConst<uint8_t,typename TA::Element>::type C;
  return Array<C>(size,(C*)data.data(),data.borrow_owner());
}

}
