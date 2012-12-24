// Instantiate necessary python conversions

#include <other/core/python/forward.h>
#include <other/core/vector/forward.h>
namespace pentago {
struct history_t;
}
namespace other {

OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_NO_EXPORT,2,uint8_t)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_NO_EXPORT,4,uint8_t)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_NO_EXPORT,3,uint64_t)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_NO_EXPORT,4,uint16_t)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_NO_EXPORT,4,Vector<uint8_t,2>)

template<> struct FromPython<pentago::history_t>{static pentago::history_t convert(PyObject* object);};
}
namespace pentago {
other::PyObject* to_python(history_t event);
}
