// Tests for parallel_for

#include "pentago/shard/parallel.h"
#include "pentago/utility/array.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
#include <atomic>
namespace pentago {
namespace {

using std::atomic;

// Every index in [0, n) is visited exactly once
TEST(parallel_for, coverage) {
  for (const int threads : {1, 2, 8}) {
    for (const size_t n : {0u, 1u, 7u, 100u, 1000u}) {
      vector<atomic<int>> counts(n);
      for (auto& c : counts) c = 0;
      parallel_for(threads, n, [&](const size_t i) {
        counts[i].fetch_add(1, std::memory_order_relaxed);
      });
      for (const size_t i : range(n))
        PENTAGO_ASSERT_EQ(counts[i].load(), 1);
    }
  }
}

// Indices are permuted (not sequential) for n > 1
TEST(parallel_for, shuffled) {
  const size_t n = 100;
  Array<size_t> order(int(n), uninit);
  atomic<int> pos(0);
  // Single thread so order is deterministic
  parallel_for(1, n, [&](const size_t i) { order[pos.fetch_add(1, std::memory_order_relaxed)] = i; });
  PENTAGO_ASSERT_EQ(size_t(pos.load()), n);
  bool any_out_of_order = false;
  for (const size_t i : range(n))
    if (order[int(i)] != i) { any_out_of_order = true; break; }
  ASSERT_TRUE(any_out_of_order);
}

}  // namespace
}  // namespace pentago
