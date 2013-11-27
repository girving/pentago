// Wrapper around C++ exceptions so that we can turn them off during MPI
#pragma once

#include <geode/utility/debug.h>
#include <geode/utility/format.h>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/is_unsigned.hpp>
#include <limits>
namespace pentago {

using namespace geode;

// Should we print?  Defaults to true.
bool verbose();
GEODE_EXPORT void set_verbose(bool verbose);

#define THROW(Error,...) \
  (pentago::maybe_throw<Error>(__VA_ARGS__))

// If nonzero, this function is called instead of throwing an exception.
GEODE_EXPORT extern ErrorCallback throw_callback;

// If nonzero, die_helper calls this function to quit
GEODE_EXPORT extern ErrorCallback die_callback;
  
// Print a message and abort without formatting
GEODE_EXPORT void GEODE_NORETURN(die_helper(const string& msg)) GEODE_COLD;

// Print a message and abort
template<class... Args> static inline void GEODE_NORETURN(die(const char* msg, const Args&... args)) GEODE_COLD;
template<class... Args> static inline void                die(const char* msg, const Args&... args) {
  die_helper(format(msg,args...));
}

namespace {
template<class T> struct assert_is_almost_uint64 {
  typedef typename boost::remove_const<T>::type I;
  static_assert(boost::is_integral<I>::value,"");
  static_assert(sizeof(I)==8,"");
  static_assert(boost::is_unsigned<I>::value,"");
};}

// Check and cast an integer to int
#define CHECK_CAST_INT(n) ({ \
  const auto _n = (n); \
  assert_is_almost_uint64<decltype(_n)>(); \
  GEODE_ASSERT(_n<=uint64_t(std::numeric_limits<int>::max())); \
  (int(_n)); })

// Everything beyond here is internals

template<class Error> GEODE_EXPORT void maybe_throw() __attribute__ ((noreturn));
template<class Error> GEODE_EXPORT void maybe_throw(const char* msg) __attribute__ ((noreturn));

template<class Error,class First,class... Rest> static inline void __attribute__ ((noreturn)) maybe_throw(const char* fmt, const First& first, const Rest&... rest) {
  maybe_throw<Error>(format(fmt,first,rest...).c_str());
}

}
