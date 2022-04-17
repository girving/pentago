// Cast a typed array down to bytes
#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/debug.h"
namespace pentago {

template<class A> static auto char_view(const A& data) {
  const int size = CHECK_CAST_INT(sizeof(typename A::value_type)*(uint64_t)data.size());
  typedef std::conditional_t<A::is_const, std::add_const_t<uint8_t>, uint8_t> C; 
  return RawArray<C>(size, reinterpret_cast<C*>(data.data()));
}

template<class A> static auto char_view_own(const A& data) {
  const int size = CHECK_CAST_INT(sizeof(typename A::value_type)*(uint64_t)data.size());
  typedef std::conditional_t<A::is_const, std::add_const_t<uint8_t>, uint8_t> C; 
  return Array<C>(vec(size), shared_ptr<C>(data.owner(), reinterpret_cast<C*>(data.data())));
}

}
