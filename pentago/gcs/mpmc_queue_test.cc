// Tests for mpmc_queue_t

#include "pentago/gcs/mpmc_queue.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
#include <atomic>
#include <thread>
namespace pentago {
namespace {

using std::atomic;
using std::thread;
using std::vector;

struct item_t {
  int value;
  int64_t bytes;
  int64_t size() const { return bytes; }
};

using Q = mpmc_queue_t<item_t>;

TEST(mpmc_queue, basic) {
  Q q(1000, 3);
  q.push({10, 100});
  q.push({20, 100});
  q.push({30, 100});

  auto a = q.pop(); ASSERT_TRUE(a.has_value()); PENTAGO_ASSERT_EQ(a->value, 10);
  auto b = q.pop(); ASSERT_TRUE(b.has_value()); PENTAGO_ASSERT_EQ(b->value, 20);
  auto c = q.pop(); ASSERT_TRUE(c.has_value()); PENTAGO_ASSERT_EQ(c->value, 30);
  ASSERT_FALSE(q.pop().has_value());
}

TEST(mpmc_queue, pop_after_total) {
  Q q(1000, 1);
  q.push({1, 10});
  ASSERT_TRUE(q.pop().has_value());
  ASSERT_FALSE(q.pop().has_value());
  ASSERT_FALSE(q.pop().has_value());
}

TEST(mpmc_queue, zero_items) {
  Q q(1000, 0);
  ASSERT_FALSE(q.pop().has_value());
}

TEST(mpmc_queue, concurrent_single_producer_single_consumer) {
  Q q(500, 100);

  thread producer([&]() {
    for (int i = 0; i < 100; i++)
      q.push({i, 10});
  });

  for (int i = 0; i < 100; i++) {
    auto item = q.pop();
    ASSERT_TRUE(item.has_value());
    PENTAGO_ASSERT_EQ(item->value, i);
  }
  ASSERT_FALSE(q.pop().has_value());
  producer.join();
}

TEST(mpmc_queue, backpressure) {
  // 100 bytes max, 50-byte items → at most 2 in queue
  Q q(100, 20);
  atomic<int> produced(0);

  thread producer([&]() {
    for (int i = 0; i < 20; i++) {
      q.push({i, 50});
      produced.fetch_add(1);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  ASSERT_LE(produced.load(), 4);

  for (int i = 0; i < 20; i++)
    ASSERT_TRUE(q.pop().has_value());
  ASSERT_FALSE(q.pop().has_value());
  producer.join();
}

TEST(mpmc_queue, multiple_producers) {
  Q q(1000, 40);
  atomic<int> next(0);

  vector<thread> producers;
  for (int t = 0; t < 4; t++)
    producers.emplace_back([&]() {
      for (;;) {
        const int i = next.fetch_add(1);
        if (i >= 40) break;
        q.push({i, 5});
      }
    });

  vector<bool> seen(40, false);
  for (int i = 0; i < 40; i++) {
    auto item = q.pop();
    ASSERT_TRUE(item.has_value());
    seen[item->value] = true;
  }
  ASSERT_FALSE(q.pop().has_value());
  for (int i = 0; i < 40; i++) ASSERT_TRUE(seen[i]);
  for (auto& t : producers) t.join();
}

TEST(mpmc_queue, multiple_consumers) {
  Q q(10000, 100);
  for (int i = 0; i < 100; i++)
    q.push({i, 10});

  atomic<int> consumed(0);
  vector<thread> consumers;
  for (int t = 0; t < 8; t++)
    consumers.emplace_back([&]() {
      for (;;) {
        if (!q.pop()) break;
        consumed.fetch_add(1);
      }
    });
  for (auto& t : consumers) t.join();
  PENTAGO_ASSERT_EQ(consumed.load(), 100);
}

TEST(mpmc_queue, concurrent_producers_and_consumers) {
  Q q(200, 200);
  atomic<int> next_prod(0);

  vector<thread> producers;
  for (int t = 0; t < 3; t++)
    producers.emplace_back([&]() {
      for (;;) {
        const int i = next_prod.fetch_add(1);
        if (i >= 200) break;
        q.push({i, 10});
      }
    });

  atomic<int> consumed(0);
  vector<thread> consumers;
  for (int t = 0; t < 8; t++)
    consumers.emplace_back([&]() {
      for (;;) {
        if (!q.pop()) break;
        consumed.fetch_add(1);
      }
    });

  for (auto& t : producers) t.join();
  for (auto& t : consumers) t.join();
  PENTAGO_ASSERT_EQ(consumed.load(), 200);
}

TEST(mpmc_queue, backpressure_with_multiple_producers_and_consumers) {
  Q q(100, 500);
  atomic<int> next_prod(0);

  vector<thread> producers;
  for (int t = 0; t < 4; t++)
    producers.emplace_back([&]() {
      for (;;) {
        const int i = next_prod.fetch_add(1);
        if (i >= 500) break;
        q.push({i, 10});
      }
    });

  atomic<int> consumed(0);
  vector<thread> consumers;
  for (int t = 0; t < 8; t++)
    consumers.emplace_back([&]() {
      for (;;) {
        if (!q.pop()) break;
        consumed.fetch_add(1);
      }
    });

  for (auto& t : producers) t.join();
  for (auto& t : consumers) t.join();
  PENTAGO_ASSERT_EQ(consumed.load(), 500);
}

}  // namespace
}  // namespace pentago
