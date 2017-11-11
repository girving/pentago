#include "pentago/data/numpy.h"
#include "pentago/utility/temporary.h"
#include "pentago/utility/log.h"
#include "pentago/utility/mmap.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

using namespace std::string_literals;

// We use regression tests so that tests don't depend on python+numpy
template<class A> void regression(const A& data, const string& expected) {
  tempdir_t tmp("npy");
  const auto path = tmp.path + "/data.npy";
  write_numpy(path, data);
  const auto contents = mmap_file(path);
  ASSERT_EQ(string(contents.begin(), contents.end()), expected);
}

template<class T, int d> void native(RawArray<const Vector<T,d>> data) {
  tempdir_t tmp("npy");
  const auto path = tmp.path + "/data.npy";
  write_numpy(path, data);
  const auto read = read_numpy<T,d>(path);
  ASSERT_EQ(data, read);
}

TEST(numpy, int) {
  const int i[] = {7, 5, -3};
  regression(asarray(i), "\x93NUMPY\x1\0F\0{'descr': '<i4', 'fortran_order': False, 'shape': (3,), }            \n\a\0\0\0\x5\0\0\0\xFD\xff\xff\xff"s);
}

TEST(numpy, vec_float) {
  const Vector<float,2> f[] = {{7.2, -3}, {8.9, 17.17}};
  regression(asarray(f), "\x93NUMPY\x1\0F\0{'descr': '<f4', 'fortran_order': False, 'shape': (2,2,), }          \nff\xE6@\0\0@\xC0"s+"ff\xE"s+"A)\\\x89"s+"A"s);
  native(asarray(f));
}

TEST(numpy, uint64) {
  const uint64_t u[] = {7237741304337476111u, 3652336030090419622u, 3374948756737560010u};
  regression(asarray(u), "\x93NUMPY\x1\0F\0{'descr': '<u8', 'fortran_order': False, 'shape': (3,), }            \n\xF*5}^\x9Fqd\xA6]\xE6\xB9"s+"D\xB4\xAF"s+"2\xCA\r}T\x3:\xD6."s);
}

}  // namespace
}  // namespace pentago
