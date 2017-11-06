#include "pentago/utility/exceptions.h"
namespace pentago {

#define INSTANTIATE(Error) \
  Error::Error(const string& message) : Base(message) {} \
  Error::~Error() throw () {}
INSTANTIATE(IOError)
INSTANTIATE(OSError)
INSTANTIATE(LookupError)
INSTANTIATE(IndexError)
INSTANTIATE(KeyError)
INSTANTIATE(TypeError)
INSTANTIATE(ValueError)
INSTANTIATE(NotImplementedError)
INSTANTIATE(AssertionError)
INSTANTIATE(AttributeError)
INSTANTIATE(ArithmeticError)
INSTANTIATE(OverflowError)
INSTANTIATE(ZeroDivisionError)
INSTANTIATE(ReferenceError)
INSTANTIATE(ImportError)

}
