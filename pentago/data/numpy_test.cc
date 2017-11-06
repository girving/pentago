#include "pentago/data/numpy.h"
#include "pentago/utility/temporary.h"
#include "pentago/utility/log.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

template<class A> string python(const A& data) {
  tempdir_t tmp("npy");
  const auto path = tmp.path + "/data.npy";
  write_numpy(path, data);
  FILE* f = popen(format("python3 -c 'import numpy; print(numpy.load(\"%s\"))'", path).c_str(), "r");
  GEODE_ASSERT(f);
  char buffer[1024];
  const int count = fread(buffer, 1, sizeof(buffer), f);
  pclose(f);
  return string(buffer, count);
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
  ASSERT_EQ(python(asarray(i)), "[ 7  5 -3]\n");
}

TEST(numpy, vec_float) {
  const Vector<float,2> f[] = {{7.2, -3}, {8.9, 17.17}};
  ASSERT_EQ(python(asarray(f)), "[[  7.19999981  -3.        ]\n [  8.89999962  17.17000008]]\n");
  native(asarray(f));
}

TEST(numpy, uint64) {
  const uint64_t u[] = {7237741304337476111u, 3652336030090419622u, 3374948756737560010u};
  ASSERT_EQ(python(asarray(u)), "[7237741304337476111 3652336030090419622 3374948756737560010]\n");
}

}  // namespace
}  // namespace pentago
