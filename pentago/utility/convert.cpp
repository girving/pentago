// Instantiate necessary python conversions
#ifdef GEODE_PYTHON

#include <pentago/utility/convert.h>
#include <pentago/base/section.h>
#include <pentago/base/superscore.h>
#include <pentago/utility/thread.h>
#include <pentago/utility/wall_time.h>
#include <geode/array/convert.h>
#include <geode/vector/convert.h>
namespace geode {

using namespace pentago;

namespace {

// Since there's no 256 bit dtype, map super_t to 4 uint64_t's
template<> struct NumpyDescr<super_t> : public NumpyDescr<uint64_t> {};
template<> struct NumpyIsStatic<super_t> : public mpl::true_ {};
template<> struct NumpyRank<super_t> : public mpl::int_<1> {};
template<> struct NumpyInfo<super_t> { static void dimensions(npy_intp* dimensions) {
  dimensions[0] = 4;
}};

// section_t is a thin wrapper around a Vector
typedef Vector<Vector<uint8_t,2>,4> CV;
template<> struct NumpyDescr<section_t> : public NumpyDescr<CV> {};
template<> struct NumpyIsStatic<section_t> : public NumpyIsStatic<CV> {};
template<> struct NumpyRank<section_t> : public NumpyRank<CV> {};
template<> struct NumpyInfo<section_t> : public NumpyInfo<CV> {};

// wall_time_t is a thin wrapper around an int64_t
template<> struct NumpyDescr<wall_time_t> : public NumpyDescr<int64_t> {};
template<> struct NumpyIsScalar<wall_time_t> : public mpl::true_ {};

// history_t looks like Vector<int64_t,3>
typedef Vector<int64_t,3> IV;
template<> struct NumpyDescr<history_t> : public NumpyDescr<IV> {};
template<> struct NumpyIsStatic<history_t> : public NumpyIsStatic<IV> {};
template<> struct NumpyRank<history_t> : public NumpyRank<IV> {};
template<> struct NumpyInfo<history_t> : public NumpyInfo<IV> {};

}

GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_NO_EXPORT,2,uint8_t)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_NO_EXPORT,4,uint8_t)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_NO_EXPORT,2,uint64_t)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_NO_EXPORT,3,uint64_t)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_NO_EXPORT,4,uint16_t)
GEODE_DEFINE_VECTOR_CONVERSIONS(GEODE_NO_EXPORT,4,Vector<uint8_t,2>)
ARRAY_CONVERSIONS(1,uint8_t)
ARRAY_CONVERSIONS(1,super_t)
ARRAY_CONVERSIONS(2,long long)
ARRAY_CONVERSIONS(4,uint64_t)
ARRAY_CONVERSIONS(4,super_t)
ARRAY_CONVERSIONS(1,Vector<uint64_t,3>)
ARRAY_CONVERSIONS(1,Vector<super_t,2>)
ARRAY_CONVERSIONS(4,Vector<super_t,2>)
NDARRAY_CONVERSIONS(section_t)
NDARRAY_CONVERSIONS(super_t)
NDARRAY_CONVERSIONS(Vector<super_t,2>)
ARRAY_CONVERSIONS(1,section_t)
ARRAY_CONVERSIONS(1,wall_time_t)
ARRAY_CONVERSIONS(1,Vector<wall_time_t,2>)
ARRAY_CONVERSIONS(1,history_t)

history_t FromPython<history_t>::convert(PyObject* object) {
  return from_numpy<history_t>(object);
}

}
namespace pentago {

PyObject* to_python(history_t event) {
  return to_numpy(event);
}

}
#endif
