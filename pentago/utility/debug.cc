// Wrapper around C++ exceptions so that we can turn them off during MPI

#include "pentago/utility/debug.h"
#include <signal.h>
#include <iostream>
namespace pentago {

static bool verbose_ = true;

bool verbose() {
  return verbose_;
}

void set_verbose(bool verbose) {
  verbose_ = verbose;
}

error_callback_t throw_callback, die_callback;

void breakpoint() {
  // If you use this you need to step out of the signal handler to get a non-corrupt stack
  raise(SIGINT);
}

void assertion_failed(const char* function, const char* file, unsigned int line,
                      const char* condition, const char* message) {
  const string error = tfm::format("%s:%d:%s: %s, condition = %s", file, line, function,
                                   message ? message : "Assertion failed", condition);
  static const bool break_on_assert = getenv("GEODE_BREAK_ON_ASSERT") != 0;
  if (break_on_assert) {
    std::cout << std::flush;
    std::cerr << "\n\n*** Error: " << error << '\n' << std::endl;
    raise(SIGINT);
  }
  THROW(AssertionError, error);
}

void die_helper(const string& msg) {
  if (die_callback)
    die_callback(msg);
  else {
    if (msg.size())
      std::cerr << "\nserial: " << msg << std::endl;
    if (getenv("GEODE_BREAK_ON_ASSERT")) breakpoint();
    exit(1);
  }
}

namespace {
template<class Error> struct error_name_t { __attribute__((unused)) static const char* name; };
}

template<class Error> void __attribute__ ((noreturn)) maybe_throw() {
  if (throw_callback)
    throw_callback(error_name_t<Error>::name);
  else
    throw Error();
}

template<class Error> void __attribute__ ((noreturn)) maybe_throw(const char* msg) {
  if (throw_callback)
    throw_callback(tfm::format("%s: %s",error_name_t<Error>::name,msg));
  else
    throw Error(msg);
}

#define REGISTER_BARE(Error) \
  namespace { template<> const char* error_name_t<Error>::name = #Error; } \
  template void __attribute__ ((noreturn)) maybe_throw<Error>();
#define REGISTER(Error) \
  namespace { template<> const char* error_name_t<Error>::name = #Error; } \
  template void __attribute__ ((noreturn)) maybe_throw<Error>(const char*);
REGISTER(AssertionError)
REGISTER(RuntimeError)
REGISTER(ValueError)
REGISTER(IOError)
REGISTER_BARE(std::bad_alloc)

}
