// Python-like exceptions
#pragma once

#include <exception>
#include <stdexcept>
#include <typeinfo>
#include <string>
namespace pentago {

#ifdef __EMSCRIPTEN__

struct RuntimeError {};
#define SIMPLE_EXCEPTION(Error, Base) struct Error {};

#else

using std::string;
using std::type_info;
using std::exception;

typedef std::runtime_error RuntimeError;

#define SIMPLE_EXCEPTION(Error, Base_) \
  struct Error : public Base_ { \
    typedef Base_ Base; \
    Error(const string& message); \
    virtual ~Error() throw (); \
  };

#endif

SIMPLE_EXCEPTION(IOError, RuntimeError)
SIMPLE_EXCEPTION(OSError, RuntimeError)
SIMPLE_EXCEPTION(LookupError, RuntimeError)
  SIMPLE_EXCEPTION(IndexError, LookupError)
  SIMPLE_EXCEPTION(KeyError, LookupError)
SIMPLE_EXCEPTION(TypeError, RuntimeError)
SIMPLE_EXCEPTION(ValueError, RuntimeError)
SIMPLE_EXCEPTION(NotImplementedError, RuntimeError)
SIMPLE_EXCEPTION(AssertionError, RuntimeError)
SIMPLE_EXCEPTION(AttributeError, RuntimeError)
SIMPLE_EXCEPTION(ArithmeticError, RuntimeError)
  SIMPLE_EXCEPTION(OverflowError, ArithmeticError)
  SIMPLE_EXCEPTION(ZeroDivisionError, ArithmeticError)
SIMPLE_EXCEPTION(ReferenceError, RuntimeError)
SIMPLE_EXCEPTION(ImportError, RuntimeError)

#undef SIMPLE_EXCEPTION

}
