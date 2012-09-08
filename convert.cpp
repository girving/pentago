// Instantiation necessary python conversions

#include <pentago/section.h>
#include <pentago/superscore.h>
#include <pentago/utility/wall_time.h>
#include <other/core/array/convert.h>
#include <other/core/vector/convert.h>
namespace other {

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

}

VECTOR_CONVERSIONS(2,uint8_t)
VECTOR_CONVERSIONS(3,uint64_t)
VECTOR_CONVERSIONS(4,uint16_t)
VECTOR_CONVERSIONS(4,Vector<uint8_t,2>)
ARRAY_CONVERSIONS(1,uint8_t)
ARRAY_CONVERSIONS(1,wall_time_t)
ARRAY_CONVERSIONS(1,super_t)
ARRAY_CONVERSIONS(4,super_t)
ARRAY_CONVERSIONS(1,Vector<uint64_t,3>)
ARRAY_CONVERSIONS(1,Vector<super_t,2>)
ARRAY_CONVERSIONS(4,Vector<super_t,2>)
NDARRAY_CONVERSIONS(section_t)
NDARRAY_CONVERSIONS(super_t)
NDARRAY_CONVERSIONS(Vector<super_t,2>)
ARRAY_CONVERSIONS(1,section_t)

}
