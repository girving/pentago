// Wrapper around C++ exceptions so that we can turn them off during MPI
//
// The pentago code does not throw during correct use.  If we see an exception,
// something terrible has happened and the entire application should be killed.
// This file implements wrappers that allow exceptions to be replaced with calls
// to MPI_Abort if any occur during parallel runs.
#pragma once

#include <type_traits>
#include <limits>
#include "pentago/utility/exceptions.h"
#include "pentago/utility/format.h"
#include "pentago/utility/wasm.h"
namespace pentago {

#ifndef __wasm__
using std::string;

// Should we print?  Defaults to true.
bool verbose();
void set_verbose(bool verbose);

#define THROW(Error, ...) \
  (pentago::maybe_throw<Error>(__VA_ARGS__))

typedef void (*error_callback_t)(const string&) __attribute__((noreturn));

// If nonzero, this function is called instead of throwing an exception.
extern error_callback_t throw_callback;

// If nonzero, die_helper calls this function to quit
extern error_callback_t die_callback;
  
// Break (for use in a debugger)
void breakpoint();

// Print a message and abort without formatting
void die_helper(const string& msg) __attribute__((noreturn, cold));

// Print a message and abort
template<class... Args> static inline void __attribute__((noreturn, cold))
die(const char* msg, const Args&... args) {
  die_helper(format(msg, args...));
}

#define GEODE_ASSERT(condition, ...) \
  ((condition) ? (void)0 : pentago::assertion_failed( \
      __PRETTY_FUNCTION__, __FILE__, __LINE__, #condition, pentago::debug_message(__VA_ARGS__)))

#else  // if __wasm__

WASM_IMPORT void __attribute__((noreturn, cold)) die(const char* msg);

#define THROW(Error, ...) die(#__VA_ARGS__)

#define STRINGIZE_DETAIL(x) #x
#define STRINGIZE(x) STRINGIZE_DETAIL(x)

#define GEODE_ASSERT(condition, ...) \
  ((condition) ? (void)0 : die(__FILE__ ":" STRINGIZE(__LINE__) ": " #condition ", " #__VA_ARGS__ ))

#endif  // __wasm__

#ifdef NDEBUG
# define GEODE_DEBUG_ONLY(...)
#else
# define GEODE_DEBUG_ONLY(...) __VA_ARGS__
#endif

// We'd use a template, but that fails for obscure reasons on clang 3.0 on Rackspace
template<class T> static inline void assert_is_almost_uint64(const T n) {
  static_assert(sizeof(T)==8,"");
  typedef std::remove_const_t<T> S;
  static_assert(   std::is_same<S,unsigned long>::value
                || std::is_same<S,unsigned long long>::value,"");
}

// Check and cast an integer to int
#define CHECK_CAST_INT(n) ({ \
  const auto _n = (n); \
  assert_is_almost_uint64(_n); \
  GEODE_ASSERT(_n<=uint64_t(std::numeric_limits<int>::max())); \
  (int(_n)); })

// Everything beyond here is internals

#ifndef __wasm__
template<class Error> void maybe_throw() __attribute__ ((noreturn, cold));
template<class Error> void maybe_throw(const char* msg) __attribute__ ((noreturn, cold));

template<class Error, class First, class... Rest> static inline void __attribute__ ((noreturn, cold))
maybe_throw(const char* fmt, const First& first, const Rest&... rest) {
#ifdef __wasm__
  maybe_throw<Error>(fmt);  // Throw away arguments for now
#else
  maybe_throw<Error>(format(fmt, first, rest...).c_str());
#endif
}

template<class Error> void __attribute__ ((noreturn, cold)) maybe_throw(const string& msg) {
  maybe_throw<Error>(msg.c_str());
}

// Helper function to work around zero-variadic argument weirdness
static inline const char* debug_message() { return 0; }
static inline const char* debug_message(const char* message) { return message; }
static inline const char* debug_message(const string& message) { return message.c_str(); }

void __attribute__((noreturn, cold))
assertion_failed(const char* function, const char* file, unsigned int line, const char* condition,
                 const char* message);
#endif  // __wasm__

}
