// Shuffled parallel_for using raw std::thread

#include "pentago/shard/parallel.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/permute.h"
#include "pentago/utility/range.h"
#include <atomic>
#include <thread>
#include <vector>
namespace pentago {

using std::atomic;
using std::thread;
using std::vector;

void parallel_for(const int num_threads, const size_t n, const std::function<void(size_t)>& f,
                  const bool sequential) {
  GEODE_ASSERT(num_threads > 0);
  const uint128_t key = 42;
  vector<thread> threads;
  threads.reserve(num_threads);
  atomic<size_t> next(0);
  for (const int t __attribute__((unused)) : range(num_threads))
    threads.emplace_back([&]() {
      for (;;) {
        const size_t i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n) break;
        f(sequential ? i : random_permute(n, key, i));
      }
    });
  for (auto& t : threads) t.join();
}

}  // namespace pentago
