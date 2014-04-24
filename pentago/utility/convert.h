// Instantiate necessary python conversions
#pragma once

#include <geode/config.h>
#ifdef GEODE_PYTHON

#include <geode/python/forward.h>
#include <geode/structure/forward.h>
#include <geode/vector/forward.h>
namespace pentago {
struct history_t;
}
namespace geode {

GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_EXPORT,2,uint8_t)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_EXPORT,4,uint8_t)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_EXPORT,2,uint64_t)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_EXPORT,3,uint64_t)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_EXPORT,4,uint16_t)
GEODE_DECLARE_VECTOR_CONVERSIONS(GEODE_EXPORT,4,Vector<uint8_t,2>)

static inline PyObject* to_python(unit) {
  Py_RETURN_NONE;
}

template<> struct FromPython<pentago::history_t>{GEODE_EXPORT static pentago::history_t convert(PyObject* object);};
}
namespace pentago {
GEODE_EXPORT geode::PyObject* to_python(history_t event);
}

#endif
