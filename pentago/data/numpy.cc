// npy i/o

#include "pentago/data/numpy.h"
#include "pentago/utility/mmap.h"
#include "pentago/utility/str.h"
#include <boost/endian/conversion.hpp>
#include <regex>
namespace pentago {

using std::min;
using std::make_tuple;

static tuple<int,char> dtype_info(const int dtype) {
  int bytes;
  char letter;
  switch (dtype) {
    #define CASE(T, dtype, let) case dtype: bytes = sizeof(T); letter = let; break;
    CASE(bool,               NPY_BOOL,       'b')
    CASE(signed char,        NPY_BYTE,       'i')
    CASE(unsigned char,      NPY_UBYTE,      'u')
    CASE(short,              NPY_SHORT,      'i')
    CASE(unsigned short,     NPY_USHORT,     'u')
    CASE(int,                NPY_INT,        'i')
    CASE(unsigned int,       NPY_UINT,       'u')
    CASE(long,               NPY_LONG,       'i')
    CASE(unsigned long,      NPY_ULONG,      'u')
    CASE(long long,          NPY_LONGLONG,   'i')
    CASE(unsigned long long, NPY_ULONGLONG,  'u')
    CASE(float,              NPY_FLOAT,      'f')
    CASE(double,             NPY_DOUBLE,     'f')
    CASE(long double,        NPY_LONGDOUBLE, 'f')
    #undef CASE
    default: throw ValueError(format("Unknown dtype %d", dtype));
  }
  return make_tuple(bytes, letter);
}

static const char endian = boost::endian::order::native == boost::endian::order::little ? '<' : '>';

tuple<string,size_t> numpy_header(RawArray<const int> shape, const int dtype) {
  const auto [bytes, letter] = dtype_info(dtype);
  string header("\x93NUMPY\x01\x00??", 10);
  header += format("{'descr': '%c%c%d', 'fortran_order': False, 'shape': (", endian, letter, bytes);
  for (const int n : shape)
    header += format("%d,", n);
  header += "), }";
  while ((header.size()+1) & 15)
    header.push_back(' ');
  header.push_back('\n');
  const auto header_size = boost::endian::native_to_little(uint16_t(header.size()-10));
  memcpy(header.data() + 8, &header_size, 2);
  return make_tuple(header, bytes*shape.product());
}

Array<const uint8_t> read_numpy_helper(const string& path, const int d, const int dtype) {
  GEODE_ASSERT(d >= 1);
  const auto [bytes, letter] = dtype_info(dtype);
  const auto data = mmap_file(path);
  const string header(data.data(), data.data() + min(1023, data.size()));
  const std::regex pattern(R"(^\x93NUMPY\x01\x00(..)\{'descr': '([<>])(.)(\d+)', 'fortran_order': (\w+), 'shape': \((\d+),\s*(\d+),\), \})");
  std::smatch m;
  if (!regex_search(header, m, pattern))
    throw ValueError(format("'%s' has an invalid header for a rank 2 .npy file", path));
  const uint16_t header_size = boost::endian::little_to_native(
      *reinterpret_cast<const uint16_t*>(m[1].str().data()));
  const char got_endian = m[2].str()[0];
  const char got_letter = m[3].str()[0];
  const string got_bytes = m[4].str();
  const string fortran_order = m[5].str();
  const string shape0 = m[6].str();
  const string shape1 = m[7].str();
  if (endian != got_endian)
    throw NotImplementedError(format("'%s' has endian %c, but native endian is %c",
                                     path, got_endian, endian));
  if (letter != got_letter || str(bytes) != got_bytes)
    throw ValueError(format("'%s' has type %c%s; we wanted type %c%d",
                            path, got_letter, got_bytes, letter, bytes));
  if (fortran_order != "False")
    throw NotImplementedError(format("'%s': fortran order %s not implemented", path, fortran_order));
  if (shape1 != str(d))
    throw ValueError(format("'%s' has shape[1] = %s, expected %d", path, shape1, d));
  char* end;
  const auto len = strtol(shape0.c_str(), &end, 0);
  if (*end || len < 0)
    throw ValueError(format("'%s' has invalid shape[0] = %s", path, shape0));
  const int expected_size = CHECK_CAST_INT(uint64_t(10) + header_size + len * d * bytes);
  if (data.size() != expected_size)
    throw ValueError(format("'%s' has size %d, expected %d", data.size(), expected_size));
  return data.slice_own(10 + header_size, data.size());
}

}  // namespace pentago
