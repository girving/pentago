// Ternary array tests

#include "pentago/data/ternary.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "pentago/utility/threefry.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

TEST(ternary, exhaustive_small) {
  for (const int n : range(16)) {
    ternaries_t t(n);
    // Verify initialized to zero
    for (const int i : range(n))
      PENTAGO_ASSERT_EQ(t[i], 0);
    // Set all combinations and verify
    for (const int v : range(3)) {
      for (const int i : range(n))
        t.set(i, v);
      for (const int i : range(n))
        PENTAGO_ASSERT_EQ(t[i], v);
    }
    // Set each position independently
    for (const int i : range(n))
      t.set(i, i % 3);
    for (const int i : range(n))
      PENTAGO_ASSERT_EQ(t[i], i % 3);
  }
}

TEST(ternary, random) {
  const auto value = [](const int i) { return int(threefry(42, i) % 3); };
  const int n = 10000;
  ternaries_t t(n);
  for (const int i : range(n))
    t.set(i, value(i));
  for (const int i : range(n))
    PENTAGO_ASSERT_EQ(t[i], value(i));
}

TEST(ternary, boundary) {
  // Sizes that are and aren't multiples of 5
  for (const int n : {0, 1, 4, 5, 6, 9, 10, 11, 99, 100, 101}) {
    ternaries_t t(n);
    for (const int i : range(n)) {
      t.set(i, 2);
      PENTAGO_ASSERT_EQ(t[i], 2);
    }
    // Verify setting one value doesn't corrupt neighbors
    for (const int i : range(n))
      t.set(i, i % 3);
    for (const int i : range(n))
      PENTAGO_ASSERT_EQ(t[i], i % 3);
  }
}

TEST(ternary, data_size) {
  PENTAGO_ASSERT_EQ(ternaries_t(0).data.size(), 0);
  PENTAGO_ASSERT_EQ(ternaries_t(1).data.size(), 1);
  PENTAGO_ASSERT_EQ(ternaries_t(5).data.size(), 1);
  PENTAGO_ASSERT_EQ(ternaries_t(6).data.size(), 2);
  PENTAGO_ASSERT_EQ(ternaries_t(10).data.size(), 2);
  PENTAGO_ASSERT_EQ(ternaries_t(11).data.size(), 3);
}

}  // namespace
}  // namespace pentago
