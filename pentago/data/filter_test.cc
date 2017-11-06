#include "pentago/data/filter.h"
#include "pentago/utility/ceil_div.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

Array<Vector<super_t,2>> random_supers(Random& random, const int n) {
  // Biased towards black wins (shouldn't matter for testing)
  Array<Vector<super_t,2>> data(n, uninit);
  for (auto& s : data) {
    s[0] = random_super(random);
    s[1] = random_super(random) & ~s[0];
  }
  return data;
}

TEST(filter, interleave) {
  Random random(8428121);
  for (int i = 0; i < 10; i++) {
    const auto src = random_supers(random, random.uniform(1000, 1200));
    const auto dst = src.copy();
    interleave(dst);
    uninterleave(dst);
    ASSERT_EQ(src, dst);
  }
}

TEST(filter, compact) {
  Random random(8428123);
  for (int i = 0; i < 10; i++) {
    const auto src = random_supers(random, random.uniform(1000, 1200));
    const auto small = compact(src.copy());
    ASSERT_EQ(small.size(), ceil_div(256*src.size(), 5));
    const auto dst = uncompact(small);
    ASSERT_EQ(src, dst);
  }
}

TEST(filter, wavelet) {
  Random random(9847224);
  for (const int s0 : {3, 8}) {
    for (const int s1 : {3, 8}) {
      for (const int s2 : {3, 8}) {
        for (const int s3 : {3, 8}) {
          const auto shape = vec(s0,s1,s2,s3);
          const auto data = random_supers(random, shape.product()).reshape_own(shape);
          const auto save = data.copy();
          wavelet_transform(data);
          wavelet_untransform(data); 
          ASSERT_EQ(data, save);
        }
      }
    }
  }
}

}  // namespace
}  // namespace pentago
