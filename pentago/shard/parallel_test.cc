// Tests for parallel_for

#include "pentago/shard/parallel.h"
#include "pentago/utility/array.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
#include <atomic>
#include <chrono>
#include <thread>
namespace pentago {
namespace {

using std::atomic;
using std::chrono::milliseconds;

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

// Every index is processed exactly once, with io result passed to compute
TEST(overlapped_parallel_for, coverage) {
  for (const int threads : {1, 2, 8}) {
    for (const size_t n : {0u, 1u, 7u, 100u}) {
      vector<atomic<int>> io_counts(n), compute_counts(n);
      for (auto& c : io_counts) c = 0;
      for (auto& c : compute_counts) c = 0;
      overlapped_parallel_for(threads, n,
        [&](const size_t i) {
          io_counts[i].fetch_add(1, std::memory_order_relaxed);
          return i * 3 + 7;  // arbitrary value to verify it's passed through
        },
        [&](const size_t i, const size_t value) {
          compute_counts[i].fetch_add(1, std::memory_order_relaxed);
          PENTAGO_ASSERT_EQ(value, i * 3 + 7);
        });
      for (const size_t i : range(n)) {
        PENTAGO_ASSERT_EQ(io_counts[i].load(), 1);
        PENTAGO_ASSERT_EQ(compute_counts[i].load(), 1);
      }
    }
  }
}

// I/O and compute actually overlap: while compute is sleeping, I/O proceeds
TEST(overlapped_parallel_for, overlap) {
  const int threads = 4;
  const size_t n = 8;
  atomic<int> active_io(0), max_concurrent_io(0);
  overlapped_parallel_for(threads, n,
    [&](const size_t) {
      const int c = active_io.fetch_add(1, std::memory_order_relaxed) + 1;
      int expected = max_concurrent_io.load(std::memory_order_relaxed);
      while (c > expected &&
             !max_concurrent_io.compare_exchange_weak(expected, c,
                 std::memory_order_relaxed, std::memory_order_relaxed)) {}
      std::this_thread::sleep_for(milliseconds(5));
      active_io.fetch_sub(1, std::memory_order_relaxed);
      return 0;
    },
    [&](const size_t, int) {
      std::this_thread::sleep_for(milliseconds(20));
    });
  // With overlap, multiple I/Os should have run concurrently
  PENTAGO_ASSERT_GT(max_concurrent_io.load(), 1);
}

}  // namespace
}  // namespace pentago
