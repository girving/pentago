#include "pentago/utility/random.h"
#include "pentago/utility/box.h"
#include "pentago/utility/log.h"
#include "pentago/utility/sqr.h"
#include "gtest/gtest.h"
#include <cmath>
namespace pentago {
namespace {

using std::abs;
using std::numeric_limits;
using std::sqrt;

template<class F> void uniform_test(const string& name, const Box<double> box, const F gen) {
  const int n = 100000;
  double sum = 0, sum_sqr = 0;
  for (int i = 0; i < n; i++) {
    const double x = gen();
    sum += x;
    sum_sqr += sqr(x);
  }
  const auto mean = sum / n;
  const auto dev = sqrt(sum_sqr / n - sqr(mean));
  const auto correct_mean = box.center();
  const auto correct_dev = sqrt(1./12 * sqr(box.shape()));
  slog("%s: %g +- %g (correct %g +- %g)", name, mean, dev, correct_mean, correct_dev);
  ASSERT_LT(abs(mean - correct_mean), 0.01 * correct_mean);
  ASSERT_LT(abs(dev - correct_dev), 0.01 * correct_dev);
}

TEST(random, float) {
  Random random(17);
  uniform_test("float", {0, 1}, [&]() { return random.uniform<float>(); });
}

TEST(random, double) {
  Random random(17);
  uniform_test("double", {0, 1}, [&]() { return random.uniform<double>(); });
}

TEST(random, bits_uint8) {
  Random random(17);
  uniform_test("bits_uint8", {0, 255}, [&]() { return random.bits<uint8_t>(); });
}

TEST(random, bits_uint32) {
  Random random(17);
  uniform_test("bits_uint32", {0, numeric_limits<uint32_t>::max()},
               [&]() { return random.bits<uint32_t>(); });
}

TEST(random, bits_uint64) {
  Random random(17);
  uniform_test("bits_uint64", {0, double(numeric_limits<uint64_t>::max())},
               [&]() { return double(random.bits<uint64_t>()); });
}

TEST(random, uniform_uint8) {
  Random random(17);
  uniform_test("uniform_uint8", {23, 87}, [&]() { return random.uniform<uint8_t>(23, 88); });
}

TEST(random, uniform_uint32) {
  Random random(17);
  uniform_test("uniform_uint32", {0, 123398232},
               [&]() { return random.uniform<uint32_t>(0, 123398232+1); });
}

TEST(random, uniform_uint64) {
  Random random(17);
  const uint64_t hi = uint64_t(1)<<40, lo = hi*2/3;
  uniform_test("uniform_uint64", {double(lo), double(hi)},
               [&]() { return double(random.uniform<uint64_t>(lo, hi)); });
}

}  // namespace
}  // namespace pentago
