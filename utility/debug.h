// Wrapper around C++ exceptions so that we can turn them off during MPI
#pragma once

#include <other/core/utility/debug.h>
#include <other/core/utility/format.h>
namespace pentago {

using namespace other;

#define THROW(Error,...) \
  (pentago::maybe_throw<Error>(__VA_ARGS__))

// If nonzero, this function is called instead of throwing an exception.
OTHER_EXPORT extern debug::ErrorCallback throw_callback;

// Everything beyond here is internals

template<class Error> OTHER_EXPORT void maybe_throw() __attribute__ ((noreturn));
template<class Error> OTHER_EXPORT void maybe_throw(const char* msg) __attribute__ ((noreturn));

template<class Error,class First,class... Rest> static inline void __attribute__ ((noreturn)) maybe_throw(const char* fmt, const First& first, const Rest&... rest) {
  maybe_throw<Error>(format(fmt,first,rest...).c_str());
}

}
