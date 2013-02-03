// Instantiate necessary python conversions
#pragma once
#ifdef OTHER_PYTHON

#include <other/core/python/forward.h>
#include <other/core/vector/forward.h>
namespace pentago {
struct history_t;
}
namespace other {

OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_EXPORT,2,uint8_t)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_EXPORT,4,uint8_t)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_EXPORT,3,uint64_t)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_EXPORT,4,uint16_t)
OTHER_DECLARE_VECTOR_CONVERSIONS(OTHER_EXPORT,4,Vector<uint8_t,2>)

template<> struct FromPython<pentago::history_t>{OTHER_EXPORT static pentago::history_t convert(PyObject* object);};
}
namespace pentago {
OTHER_EXPORT other::PyObject* to_python(history_t event);
}

#endif
