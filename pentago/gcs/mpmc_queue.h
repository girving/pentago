// Thread-safe bounded multi-producer multi-consumer queue
//
// push() blocks when the queue exceeds max_bytes (backpressure).
// pop() blocks when empty, returns nullopt after total_items have been popped.
// T must have a size() method returning int or int64_t.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
namespace pentago {

using std::optional;

template<class T>
class mpmc_queue_t {
  const int64_t max_bytes;

  std::mutex mu;
  std::condition_variable push_cv;  // producers wait for space
  std::condition_variable pop_cv;   // consumers wait for items
  std::deque<T> items;
  int64_t current_bytes = 0;
  std::atomic<int64_t> unclaimed;

public:
  mpmc_queue_t(const int64_t max_bytes, const int64_t total_items)
    : max_bytes(max_bytes), unclaimed(total_items) {}

  void push(T item) {
    std::unique_lock<std::mutex> lock(mu);
    push_cv.wait(lock, [&]() { return current_bytes < max_bytes; });
    current_bytes += item.size();
    items.push_back(std::move(item));
    pop_cv.notify_one();
  }

  optional<T> pop() {
    // Claim a slot without holding the mutex
    if (unclaimed.fetch_sub(1) <= 0) return {};
    std::unique_lock<std::mutex> lock(mu);
    pop_cv.wait(lock, [&]() { return !items.empty(); });
    T item = std::move(items.front());
    items.pop_front();
    current_bytes -= item.size();
    push_cv.notify_one();
    return item;
  }
};

}  // namespace pentago
