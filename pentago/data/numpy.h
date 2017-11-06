// npy i/o
#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/char_view.h"
#include "pentago/utility/scalar_view.h"
namespace pentago {

using std::tuple;

// Lifted from numpy/ndarraytypes.h
enum NPY_TYPES { NPY_BOOL=0,
                 NPY_BYTE, NPY_UBYTE,
                 NPY_SHORT, NPY_USHORT,
                 NPY_INT, NPY_UINT,
                 NPY_LONG, NPY_ULONG,
                 NPY_LONGLONG, NPY_ULONGLONG,
                 NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                 NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
                 NPY_OBJECT=17,
                 NPY_STRING, NPY_UNICODE,
                 NPY_VOID
};

template<class T> struct numpy_dtype;
#define DTYPE(T, val) template<> struct numpy_dtype<T> { static const int value = val; };
DTYPE(bool, NPY_BOOL)
DTYPE(signed char, NPY_BYTE)
DTYPE(unsigned char, NPY_UBYTE)
DTYPE(short, NPY_SHORT)
DTYPE(unsigned short, NPY_USHORT)
DTYPE(int, NPY_INT)
DTYPE(unsigned int, NPY_UINT)
DTYPE(long, NPY_LONG)
DTYPE(unsigned long, NPY_ULONG)
DTYPE(long long, NPY_LONGLONG)
DTYPE(unsigned long long, NPY_ULONGLONG)
DTYPE(float, NPY_FLOAT)
DTYPE(double, NPY_DOUBLE)
DTYPE(long double, NPY_LONGDOUBLE)
#undef DTYPE

// (header,data_size) for an .npy file
tuple<string,size_t> numpy_header(RawArray<const int> shape, const int dtype);

template<class A> tuple<string,size_t> numpy_header(const A& data) {
  const auto scalars = scalar_view(data);
  typedef typename decltype(scalars)::value_type scalar;
  return numpy_header(asarray(scalars.shape()), numpy_dtype<scalar>::value);
}

// Write an .npy file
template<class A> void write_numpy(const string& filename, const A& data) {
  const auto [header, size] = numpy_header(data);
  const auto chars = char_view(data);
  GEODE_ASSERT(size == chars.size());

  // Write npy file
  FILE* file = fopen(filename.c_str(), "wb");
  if (!file) throw OSError(format("Can't open %s for writing", filename));
  fwrite(header.data(), 1, header.size(), file);
  fwrite(chars.data(), 1, size, file);
  fclose(file);
}

// Mmap an .npy file and return a flat char view
Array<const uint8_t> read_numpy_helper(const string& filename, const int d, const int dtype);

// Mmap an .npy file.  Restricted to rank 2 with last dim known for now.
template<class T, int d> Array<const Vector<T,d>> read_numpy(const string& filename) {
  const auto chars = read_numpy_helper(filename, d, numpy_dtype<T>::value);
  return Array<const Vector<T,d>>(
      vec(chars.size() / (d*int(sizeof(T)))),
      shared_ptr<const Vector<T,d>>(
          chars.owner(), reinterpret_cast<const Vector<T,d>*>(chars.data())));
}

}
