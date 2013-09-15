// Wrapper around C++ exceptions so that we can turn them off during MPI

#include <pentago/utility/debug.h>
#include <other/core/python/exceptions.h>
#include <other/core/utility/process.h>
#include <iostream>
namespace pentago {

using std::cerr;
using std::endl;
using std::bad_alloc;

static bool verbose_ = true;

bool verbose() {
  return verbose_;
}

void set_verbose(bool verbose) {
  verbose_ = verbose;
}

ErrorCallback throw_callback, die_callback;

void die_helper(const string& msg) {
  if (die_callback)
    die_callback(msg);
  else {
    cerr << "\nserial: " << msg << endl;
    process::backtrace();
    if (getenv("OTHER_BREAK_ON_ASSERT"))
      breakpoint();
    exit(1);
  }
}

namespace {
template<class Error> struct error_name_t { static const char* name; };
}

template<class Error> void __attribute__ ((noreturn)) maybe_throw() {
  if (throw_callback)
    throw_callback(error_name_t<Error>::name);
  else
    throw Error();
}

template<class Error> void __attribute__ ((noreturn)) maybe_throw(const char* msg) {
  if (throw_callback)
    throw_callback(format("%s: %s",error_name_t<Error>::name,msg));
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
REGISTER_BARE(bad_alloc)

}
