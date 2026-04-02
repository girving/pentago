// Ternary array tests

#include "pentago/data/ternary.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "pentago/utility/thread.h"
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

TEST(ternary, byte_count) {
  PENTAGO_ASSERT_EQ(ternaries_t(0).bytes().size(), 0);
  PENTAGO_ASSERT_EQ(ternaries_t(1).bytes().size(), 1);
  PENTAGO_ASSERT_EQ(ternaries_t(5).bytes().size(), 1);
  PENTAGO_ASSERT_EQ(ternaries_t(6).bytes().size(), 2);
  PENTAGO_ASSERT_EQ(ternaries_t(10).bytes().size(), 2);
  PENTAGO_ASSERT_EQ(ternaries_t(11).bytes().size(), 3);
}

TEST(ternary, counts) {
  // Uniform
  const auto value = [](const int i) { return int(threefry(42, i) % 3); };
  const int n = 10000;
  ternaries_t t(n);
  uint64_t expected[3] = {};
  for (const int i : range(n)) {
    const int v = value(i);
    t.set(i, v);
    expected[v]++;
  }
  const auto c = t.counts();
  PENTAGO_ASSERT_EQ(c[0], expected[0]);
  PENTAGO_ASSERT_EQ(c[1], expected[1]);
  PENTAGO_ASSERT_EQ(c[2], expected[2]);

  // All same
  for (const int v : {0, 1, 2}) {
    ternaries_t t2(100);
    for (const int i : range(100)) t2.set(i, v);
    const auto c2 = t2.counts();
    PENTAGO_ASSERT_EQ(c2[v], 100);
    PENTAGO_ASSERT_EQ(c2[(v+1)%3], 0);
    PENTAGO_ASSERT_EQ(c2[(v+2)%3], 0);
  }

  // Empty
  PENTAGO_ASSERT_EQ(ternaries_t(0).counts(), vec<uint64_t>(0, 0, 0));

  // Sizes not multiples of 5
  for (const int n2 : {1, 2, 3, 4, 6, 7, 8, 9, 11}) {
    ternaries_t t3(n2);
    for (const int i : range(n2)) t3.set(i, i % 3);
    const auto c3 = t3.counts();
    PENTAGO_ASSERT_EQ(c3[0] + c3[1] + c3[2], n2);
  }
}

TEST(ternary, fill_random) {
  const auto thresh = Vector<uint16_t,2>(uint16_t(0.6 * 65536), uint16_t(0.9 * 65536));
  // Test all sizes 0..50 plus some larger ones
  for (const int n : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 39, 40, 41, 100, 1000}) {
    ternaries_t t(n);
    t.fill_random(42, thresh);
    // Verify all values are in {0,1,2}
    for (const int i : range(n))
      ASSERT_LT(t[i], 3) << "n=" << n << " i=" << i;
    // Verify counts sum to n
    const auto c = t.counts();
    PENTAGO_ASSERT_EQ(c[0] + c[1] + c[2], n);
  }
  // Verify determinism
  ternaries_t a(1000), b(1000);
  a.fill_random(42, thresh);
  b.fill_random(42, thresh);
  for (const int i : range(1000))
    PENTAGO_ASSERT_EQ(a[i], b[i]);
  // Verify different keys give different results
  ternaries_t c(1000);
  c.fill_random(43, thresh);
  int diffs = 0;
  for (const int i : range(1000))
    diffs += a[i] != c[i];
  ASSERT_GT(diffs, 100);
}

TEST(ternary, atomic_set_from_zero) {
  // Sequential: verify it produces the same result as set() on zero-initialized buffers
  for (const int n : {0, 1, 4, 5, 6, 10, 11, 100, 101, 1000}) {
    ternaries_t a(n), b(n);
    for (const int i : range(n)) {
      const int v = int(threefry(77, i) % 3);
      a.set(i, v);
      b.atomic_set_from_zero(i, v);
    }
    for (const int i : range(n))
      PENTAGO_ASSERT_EQ(a[i], b[i]);
  }

  // Concurrent: many threads writing disjoint positions
  init_threads(-1, -1);
  {
    const int n = 100000;
    ternaries_t t(n);
    for (const int i : range(n)) {
      threads_schedule(CPU, [&, i]() {
        t.atomic_set_from_zero(i, int(threefry(88, i) % 3));
      });
    }
    threads_wait_all();
    for (const int i : range(n))
      PENTAGO_ASSERT_EQ(t[i], int(threefry(88, i) % 3));
  }

  // Concurrent: multiple threads writing to the same byte (positions sharing a group of 5)
  {
    const int groups = 20000;
    ternaries_t t(groups * 5);
    for (const int g : range(groups)) {
      threads_schedule(CPU, [&, g]() {
        for (const int d : range(5)) {
          const int v = int(threefry(99, g * 5 + d) % 3);
          t.atomic_set_from_zero(g * 5 + d, v);
        }
      });
    }
    threads_wait_all();
    for (const int i : range(groups * 5))
      PENTAGO_ASSERT_EQ(t[i], int(threefry(99, i) % 3));
  }
}

}  // namespace
}  // namespace pentago
