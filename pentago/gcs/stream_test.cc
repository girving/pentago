// Tests for streamer_t

#include "pentago/gcs/stream.h"
#include "pentago/utility/array.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
#include <atomic>
#include <thread>
namespace pentago {
namespace {

using std::atomic;
using std::thread;
using std::vector;

// Helper: create a fake file as a byte array with deterministic content
static uint8_t fake_byte(const int64_t pos) { return uint8_t(pos * 31 + 17); }

static fetch_fn_t make_fake_fetch(const int64_t file_size, atomic<int>& fetch_count) {
  return [file_size, &fetch_count](const int64_t offset, const int64_t size) -> Array<const uint8_t> {
    fetch_count.fetch_add(1);
    if (offset >= file_size) return {};
    const int64_t actual = std::min(size, file_size - offset);
    Array<uint8_t> data(int(actual), uninit);
    for (int i = 0; i < int(actual); i++)
      data[i] = fake_byte(offset + i);
    return data;
  };
}

TEST(stream, basic) {
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(1000, fetches);

  // Request 3 non-overlapping ranges
  const streamer_t::request_t reqs[] = {{10, 5, 0}, {500, 10, 1}, {990, 10, 2}};
  streamer_t s(fetch, asarray(reqs), /*chunk_bytes=*/100, /*readahead_bytes=*/1000, /*num_threads=*/2);

  for (int i = 0; i < 3; i++) {
    auto r = s.next();
    ASSERT_TRUE(bool(r));
    PENTAGO_ASSERT_EQ(r.id, i);
    PENTAGO_ASSERT_EQ(r.data.size(), reqs[i].size);
    for (int j = 0; j < reqs[i].size; j++)
      PENTAGO_ASSERT_EQ(r.data[j], fake_byte(reqs[i].offset + j));
  }
  ASSERT_FALSE(bool(s.next()));
}

TEST(stream, unsorted_requests) {
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(1000, fetches);

  // Submit requests out of order — streamer sorts internally
  const streamer_t::request_t reqs[] = {{900, 10, 2}, {100, 10, 0}, {500, 10, 1}};
  streamer_t s(fetch, asarray(reqs), 100, 1000, 2);

  // Results come in offset order: 100, 500, 900
  auto r0 = s.next(); ASSERT_TRUE(bool(r0)); PENTAGO_ASSERT_EQ(r0.id, 0);
  auto r1 = s.next(); ASSERT_TRUE(bool(r1)); PENTAGO_ASSERT_EQ(r1.id, 1);
  auto r2 = s.next(); ASSERT_TRUE(bool(r2)); PENTAGO_ASSERT_EQ(r2.id, 2);
  ASSERT_FALSE(bool(s.next()));
}

TEST(stream, request_near_chunk_boundary) {
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(200, fetches);

  // Request near chunk boundary but within one chunk (chunks are 50 bytes)
  const streamer_t::request_t reqs[] = {{30, 20, 0}, {60, 20, 1}};
  streamer_t s(fetch, asarray(reqs), 50, 1000, 1);

  auto r0 = s.next();
  ASSERT_TRUE(bool(r0));
  PENTAGO_ASSERT_EQ(r0.data.size(), 20);
  for (int i = 0; i < 20; i++)
    PENTAGO_ASSERT_EQ(r0.data[i], fake_byte(30 + i));

  auto r1 = s.next();
  ASSERT_TRUE(bool(r1));
  for (int i = 0; i < 20; i++)
    PENTAGO_ASSERT_EQ(r1.data[i], fake_byte(60 + i));
}

TEST(stream, small_file) {
  atomic<int> fetches(0);
  // File is smaller than one chunk
  const auto fetch = make_fake_fetch(30, fetches);

  const streamer_t::request_t reqs[] = {{0, 10, 0}, {20, 10, 1}};
  streamer_t s(fetch, asarray(reqs), 100, 1000, 4);

  auto r0 = s.next(); ASSERT_TRUE(bool(r0));
  PENTAGO_ASSERT_EQ(r0.data.size(), 10);
  for (int i = 0; i < 10; i++) PENTAGO_ASSERT_EQ(r0.data[i], fake_byte(i));

  auto r1 = s.next(); ASSERT_TRUE(bool(r1));
  PENTAGO_ASSERT_EQ(r1.data.size(), 10);
  for (int i = 0; i < 10; i++) PENTAGO_ASSERT_EQ(r1.data[i], fake_byte(20 + i));

  ASSERT_FALSE(bool(s.next()));
}

TEST(stream, many_small_requests) {
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(500, fetches);

  // 50 requests of 5 bytes each, scattered across the file
  vector<streamer_t::request_t> reqs;
  for (int i = 0; i < 50; i++)
    reqs.push_back({int64_t(i * 10), 5, i});
  streamer_t s(fetch, asarray(reqs), 50, 200, 3);

  for (int i = 0; i < 50; i++) {
    auto r = s.next();
    ASSERT_TRUE(bool(r));
    PENTAGO_ASSERT_EQ(r.data.size(), 5);
    const int64_t offset = r.id * 10;
    for (int j = 0; j < 5; j++)
      PENTAGO_ASSERT_EQ(r.data[j], fake_byte(offset + j));
  }
  ASSERT_FALSE(bool(s.next()));
}

TEST(stream, concurrent_consumers) {
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(10000, fetches);

  vector<streamer_t::request_t> reqs;
  for (int i = 0; i < 100; i++)
    reqs.push_back({int64_t(i * 100), 50, i});
  streamer_t s(fetch, asarray(reqs), 200, 2000, 3);

  // 8 consumer threads, each popping until done
  atomic<int> consumed(0);
  vector<thread> consumers;
  for (int t = 0; t < 8; t++)
    consumers.emplace_back([&]() {
      for (;;) {
        auto r = s.next();
        if (!r) break;
        const int64_t offset = r.id * 100;
        for (int j = 0; j < 50; j++)
          PENTAGO_ASSERT_EQ(r.data[j], fake_byte(offset + j));
        consumed.fetch_add(1);
      }
    });
  for (auto& t : consumers) t.join();
  PENTAGO_ASSERT_EQ(consumed.load(), 100);
}

TEST(stream, backpressure) {
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(10000, fetches);

  // Small readahead (200 bytes = 2 chunks of 100) to test backpressure
  vector<streamer_t::request_t> reqs;
  for (int i = 0; i < 50; i++)
    reqs.push_back({int64_t(i * 200), 10, i});
  streamer_t s(fetch, asarray(reqs), 100, 200, 2);

  for (int i = 0; i < 50; i++) {
    auto r = s.next();
    ASSERT_TRUE(bool(r));
    PENTAGO_ASSERT_EQ(r.data.size(), 10);
  }
  ASSERT_FALSE(bool(s.next()));
}

TEST(stream, grouping) {
  // With chunk_bytes=100, requests spanning 250 bytes should produce 3 groups
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(1000, fetches);

  const streamer_t::request_t reqs[] = {
    {0, 10, 0}, {20, 10, 1}, {50, 10, 2},     // group 1: 0-99 (span 60, < 100)...
    {90, 10, 3},                                // group 1: 0-99 (span 100, closes)
    {150, 10, 4}, {200, 10, 5}, {260, 10, 6},  // group 2: 150-269 (span 120, closes at 270)
    {400, 10, 7},                               // group 3: 400-409 (final)
  };
  streamer_t s(fetch, asarray(reqs), 100, 10000, 1);

  // Should get all 8 results with correct data
  vector<bool> seen(8, false);
  for (int i = 0; i < 8; i++) {
    auto r = s.next();
    ASSERT_TRUE(bool(r));
    seen[r.id] = true;
    for (int j = 0; j < 10; j++)
      PENTAGO_ASSERT_EQ(r.data[j], fake_byte(reqs[r.id].offset + j));
  }
  for (int i = 0; i < 8; i++) ASSERT_TRUE(seen[i]);
  ASSERT_FALSE(bool(s.next()));
  // 3 groups → 3 fetches
  PENTAGO_ASSERT_EQ(fetches.load(), 3);
}

TEST(stream, high_offset) {
  // Requests not starting at offset 0
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(10000, fetches);

  const streamer_t::request_t reqs[] = {{5000, 10, 0}, {5020, 10, 1}, {5050, 10, 2}};
  streamer_t s(fetch, asarray(reqs), 100, 10000, 1);

  for (int i = 0; i < 3; i++) {
    auto r = s.next();
    ASSERT_TRUE(bool(r));
    for (int j = 0; j < 10; j++)
      PENTAGO_ASSERT_EQ(r.data[j], fake_byte(reqs[r.id].offset + j));
  }
  ASSERT_FALSE(bool(s.next()));
}

TEST(stream, overlapping_rejected) {
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(1000, fetches);

  // Overlapping requests: [10,20) and [15,25)
  const streamer_t::request_t reqs[] = {{10, 10, 0}, {15, 10, 1}};
  ASSERT_ANY_THROW(streamer_t(fetch, asarray(reqs), 100, 1000, 1));
}

TEST(stream, single_large_request) {
  // One request that is exactly chunk_bytes
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(1000, fetches);

  const streamer_t::request_t reqs[] = {{100, 50, 0}};
  streamer_t s(fetch, asarray(reqs), 50, 1000, 1);

  auto r = s.next();
  ASSERT_TRUE(bool(r));
  PENTAGO_ASSERT_EQ(r.data.size(), 50);
  for (int i = 0; i < 50; i++)
    PENTAGO_ASSERT_EQ(r.data[i], fake_byte(100 + i));
  ASSERT_FALSE(bool(s.next()));
}

TEST(stream, empty_requests) {
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(100, fetches);

  streamer_t s(fetch, RawArray<const streamer_t::request_t>(), 100, 1000, 1);
  ASSERT_FALSE(bool(s.next()));
}

TEST(stream, early_destruction) {
  // Simulate dry-run: many requests, but destroy streamer after consuming only a few.
  // Without cancel(), reader threads block in push() and destructor deadlocks on join().
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(100000, fetches);

  vector<streamer_t::request_t> reqs;
  for (int i = 0; i < 1000; i++)
    reqs.push_back({int64_t(i * 100), 50, i});

  // Small readahead to ensure producers block quickly
  streamer_t s(fetch, asarray(reqs), 200, 500, 4);

  // Consume only 3 of 1000, then let destructor run
  for (int i = 0; i < 3; i++) {
    auto r = s.next();
    ASSERT_TRUE(bool(r));
  }
  // Destructor runs here — must not deadlock
}

TEST(stream, early_destruction_no_consume) {
  // Extreme case: construct streamer but never call next()
  atomic<int> fetches(0);
  const auto fetch = make_fake_fetch(10000, fetches);

  vector<streamer_t::request_t> reqs;
  for (int i = 0; i < 100; i++)
    reqs.push_back({int64_t(i * 100), 50, i});

  streamer_t s(fetch, asarray(reqs), 200, 200, 4);
  // Destructor runs here — must not deadlock
}

}  // namespace
}  // namespace pentago
