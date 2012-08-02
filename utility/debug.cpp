// Wrapper around C++ exceptions so that we can turn them off during MPI

#include <pentago/utility/debug.h>
#include <other/core/python/Exceptions.h>
namespace pentago {

using std::bad_alloc;

debug::ErrorCallback throw_callback;

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
  template void maybe_throw<Error>() __attribute__ ((noreturn));
#define REGISTER(Error) \
  namespace { template<> const char* error_name_t<Error>::name = #Error; } \
  template void maybe_throw<Error>(const char*) __attribute__ ((noreturn));
REGISTER(AssertionError)
REGISTER(RuntimeError)
REGISTER(ValueError)
REGISTER(IOError)
REGISTER_BARE(bad_alloc)

}
