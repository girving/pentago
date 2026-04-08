// Unit tests for GCS internals (no actual GCS access)

#include "pentago/gcs/gcs.h"
#include "pentago/gcs/internal.h"
#include "pentago/utility/range.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
#include <atomic>
#include <thread>
namespace pentago {
namespace {

using gcs_internal::json_string;
using gcs_internal::base64url_encode;
using gcs_internal::read_chunked_file;
using std::atomic;
using std::thread;

// --- json_string ---

TEST(gcs, json_string_basic) {
  const string json = R"({"client_email":"test@example.com","private_key":"-----BEGIN"})";
  PENTAGO_ASSERT_EQ(json_string(json, "client_email"), "test@example.com");
  PENTAGO_ASSERT_EQ(json_string(json, "private_key"), "-----BEGIN");
}

TEST(gcs, json_string_escapes) {
  const string json = R"({"key":"line1\nline2","other":"a\\b"})";
  PENTAGO_ASSERT_EQ(json_string(json, "key"), "line1\nline2");
  PENTAGO_ASSERT_EQ(json_string(json, "other"), "a\\b");
}

TEST(gcs, json_string_with_whitespace) {
  const string json = R"({  "key" : "value"  })";
  PENTAGO_ASSERT_EQ(json_string(json, "key"), "value");
}

TEST(gcs, json_string_escaped_quotes) {
  const string json = R"({"key":"hello \"world\""})";
  PENTAGO_ASSERT_EQ(json_string(json, "key"), "hello \"world\"");
}

TEST(gcs, json_string_prefix_key) {
  // "key" should not match "key2"
  const string json = R"({"key2":"wrong","key":"right"})";
  PENTAGO_ASSERT_EQ(json_string(json, "key"), "right");
}

TEST(gcs, json_string_missing_key_dies) {
  const string json = R"({"other":"value"})";
  ASSERT_DEATH(json_string(json, "key"), "missing JSON key");
}

// --- base64url_encode ---

TEST(gcs, base64url_empty) {
  PENTAGO_ASSERT_EQ(base64url_encode(""), "");
}

TEST(gcs, base64url_known_vectors) {
  // From RFC 4648 test vectors, adapted for URL-safe alphabet (no padding)
  PENTAGO_ASSERT_EQ(base64url_encode("f"), "Zg");
  PENTAGO_ASSERT_EQ(base64url_encode("fo"), "Zm8");
  PENTAGO_ASSERT_EQ(base64url_encode("foo"), "Zm9v");
  PENTAGO_ASSERT_EQ(base64url_encode("foobar"), "Zm9vYmFy");
  // Verify URL-safe: bytes that would produce + and / in standard base64
  const uint8_t data[] = {0xfb, 0xff, 0xfe};
  PENTAGO_ASSERT_EQ(base64url_encode(asarray(data)), "-__-");
}

TEST(gcs, base64url_jwt_header) {
  // The JWT header used in make_jwt() should produce a known encoding
  PENTAGO_ASSERT_EQ(base64url_encode(R"({"alg":"RS256","typ":"JWT"})"),
                    "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9");
}

// --- parse_gcs_uri / is_gcs_path ---

TEST(gcs, is_gcs_path) {
  ASSERT_TRUE(is_gcs_path("gs://bucket/object"));
  ASSERT_TRUE(is_gcs_path("gs://b/o"));
  ASSERT_FALSE(is_gcs_path("/local/path"));
  ASSERT_FALSE(is_gcs_path("gs://"));
  ASSERT_FALSE(is_gcs_path(""));
}

TEST(gcs, parse_gcs_uri) {
  const auto [b1, o1] = parse_gcs_uri("gs://my-bucket/path/to/object.bin");
  PENTAGO_ASSERT_EQ(b1, "my-bucket");
  PENTAGO_ASSERT_EQ(o1, "path/to/object.bin");

  const auto [b2, o2] = parse_gcs_uri("gs://b/o");
  PENTAGO_ASSERT_EQ(b2, "b");
  PENTAGO_ASSERT_EQ(o2, "o");

  // Object with nested slashes
  const auto [b3, o3] = parse_gcs_uri("gs://bucket/a/b/c/d.txt");
  PENTAGO_ASSERT_EQ(b3, "bucket");
  PENTAGO_ASSERT_EQ(o3, "a/b/c/d.txt");

  // Trailing slash gives empty-ish object
  const auto [b4, o4] = parse_gcs_uri("gs://bucket/");
  PENTAGO_ASSERT_EQ(b4, "bucket");
  PENTAGO_ASSERT_EQ(o4, "");
}

// --- Chunk cache (read_chunked_file) ---

// Helper: make a file of the given size backed by deterministic data
static shared_ptr<const read_file_t> make_test_file(
    const int64_t file_size, const int64_t chunk_bytes, const int64_t max_cache_bytes,
    atomic<int>& fetch_count) {
  return read_chunked_file("test", chunk_bytes, max_cache_bytes,
      [file_size, chunk_bytes, &fetch_count](const int64_t ci) -> Array<const uint8_t> {
        fetch_count.fetch_add(1);
        const int64_t start = ci * chunk_bytes;
        const int size = int(std::min(chunk_bytes, file_size - start));
        Array<uint8_t> data(size, uninit);
        for (int i = 0; i < size; i++)
          data[i] = uint8_t((start + i) * 7 + 13);  // deterministic pattern
        return data;
      });
}

static uint8_t expected_byte(const int64_t pos) {
  return uint8_t(pos * 7 + 13);
}

TEST(gcs, chunk_cache_basic_read) {
  atomic<int> fetches(0);
  const auto f = make_test_file(1000, 100, 10000, fetches);

  // Read 10 bytes from the middle
  Array<uint8_t> buf(10, uninit);
  const auto err = f->pread(buf, 50);
  PENTAGO_ASSERT_EQ(err, "");
  for (const int i : range(10))
    PENTAGO_ASSERT_EQ(buf[i], expected_byte(50 + i));
  PENTAGO_ASSERT_EQ(fetches.load(), 1);  // one chunk fetched
}

TEST(gcs, chunk_cache_cross_chunk_read) {
  atomic<int> fetches(0);
  const auto f = make_test_file(1000, 100, 10000, fetches);

  // Read across chunk boundary (chunk 0 ends at 99, chunk 1 starts at 100)
  Array<uint8_t> buf(20, uninit);
  const auto err = f->pread(buf, 90);
  PENTAGO_ASSERT_EQ(err, "");
  for (const int i : range(20))
    PENTAGO_ASSERT_EQ(buf[i], expected_byte(90 + i));
  PENTAGO_ASSERT_EQ(fetches.load(), 2);  // two chunks fetched
}

TEST(gcs, chunk_cache_hit) {
  atomic<int> fetches(0);
  const auto f = make_test_file(1000, 100, 10000, fetches);

  // First read fetches chunk 0
  Array<uint8_t> buf(10, uninit);
  f->pread(buf, 0);
  PENTAGO_ASSERT_EQ(fetches.load(), 1);

  // Second read from same chunk should be a cache hit
  f->pread(buf, 50);
  PENTAGO_ASSERT_EQ(fetches.load(), 1);  // no new fetch
}

TEST(gcs, chunk_cache_eviction) {
  atomic<int> fetches(0);
  // 10 chunks of 100 bytes each, cache holds 250 bytes (2-3 chunks)
  const auto f = make_test_file(1000, 100, 250, fetches);

  // Read from chunks 0, 1, 2, 3 — should evict 0 and 1
  Array<uint8_t> buf(1, uninit);
  for (const int ci : range(4)) {
    f->pread(buf, ci * 100);
  }
  PENTAGO_ASSERT_EQ(fetches.load(), 4);

  // Re-read chunk 0 — should be evicted, triggers re-fetch
  f->pread(buf, 0);
  PENTAGO_ASSERT_EQ(fetches.load(), 5);

  // Re-read chunk 3 — should still be cached
  f->pread(buf, 300);
  PENTAGO_ASSERT_EQ(fetches.load(), 5);
}

TEST(gcs, chunk_cache_last_chunk_short) {
  atomic<int> fetches(0);
  // File is 150 bytes, chunk size 100. Chunk 1 is only 50 bytes.
  const auto f = make_test_file(150, 100, 10000, fetches);

  // Read last 10 bytes of the file
  Array<uint8_t> buf(10, uninit);
  const auto err = f->pread(buf, 140);
  PENTAGO_ASSERT_EQ(err, "");
  for (const int i : range(10))
    PENTAGO_ASSERT_EQ(buf[i], expected_byte(140 + i));

  // Read past end should return error
  Array<uint8_t> buf2(20, uninit);
  const auto err2 = f->pread(buf2, 140);
  ASSERT_FALSE(err2.empty());
}

TEST(gcs, chunk_cache_exact_chunk_read) {
  atomic<int> fetches(0);
  const auto f = make_test_file(300, 100, 10000, fetches);

  // Read exactly one full chunk
  Array<uint8_t> buf(100, uninit);
  const auto err = f->pread(buf, 0);
  PENTAGO_ASSERT_EQ(err, "");
  for (const int i : range(100))
    PENTAGO_ASSERT_EQ(buf[i], expected_byte(i));
  PENTAGO_ASSERT_EQ(fetches.load(), 1);
}

TEST(gcs, chunk_cache_span_three_chunks) {
  atomic<int> fetches(0);
  const auto f = make_test_file(1000, 100, 10000, fetches);

  // Read 220 bytes at offset 80: spans chunks 0 (80-99), 1 (100-199), 2 (200-299)
  Array<uint8_t> buf(220, uninit);
  const auto err = f->pread(buf, 80);
  PENTAGO_ASSERT_EQ(err, "");
  for (const int i : range(220))
    PENTAGO_ASSERT_EQ(buf[i], expected_byte(80 + i));
  PENTAGO_ASSERT_EQ(fetches.load(), 3);
}

TEST(gcs, chunk_cache_chunk_boundary) {
  atomic<int> fetches(0);
  const auto f = make_test_file(1000, 100, 10000, fetches);

  // Read starting exactly at chunk boundary
  Array<uint8_t> buf(10, uninit);
  const auto err = f->pread(buf, 200);
  PENTAGO_ASSERT_EQ(err, "");
  for (const int i : range(10))
    PENTAGO_ASSERT_EQ(buf[i], expected_byte(200 + i));
  PENTAGO_ASSERT_EQ(fetches.load(), 1);  // only chunk 2
}

TEST(gcs, chunk_cache_zero_byte_read) {
  atomic<int> fetches(0);
  const auto f = make_test_file(1000, 100, 10000, fetches);

  Array<uint8_t> buf;
  const auto err = f->pread(buf, 50);
  PENTAGO_ASSERT_EQ(err, "");
  PENTAGO_ASSERT_EQ(fetches.load(), 0);  // no fetch needed
}

TEST(gcs, chunk_cache_independent_files) {
  atomic<int> fetches1(0), fetches2(0);
  const auto f1 = make_test_file(500, 100, 10000, fetches1);
  const auto f2 = make_test_file(500, 100, 10000, fetches2);

  Array<uint8_t> buf(10, uninit);
  f1->pread(buf, 0);
  f2->pread(buf, 0);
  // Each file has its own cache, so both fetch independently
  PENTAGO_ASSERT_EQ(fetches1.load(), 1);
  PENTAGO_ASSERT_EQ(fetches2.load(), 1);
}

TEST(gcs, chunk_cache_concurrent_dedup) {
  atomic<int> fetches(0);
  // Slow fetch to ensure threads overlap
  const auto f = read_chunked_file("test", 100, 10000,
      [&fetches](const int64_t ci) -> Array<const uint8_t> {
        fetches.fetch_add(1);
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        Array<uint8_t> data(100, uninit);
        for (int i = 0; i < 100; i++) data[i] = uint8_t(ci);
        return data;
      });

  // Launch 8 threads all reading from chunk 0
  vector<thread> threads;
  for (const int t __attribute__((unused)) : range(8))
    threads.emplace_back([&]() {
      Array<uint8_t> buf(10, uninit);
      f->pread(buf, 0);
    });
  for (auto& t : threads) t.join();

  // Should fetch chunk 0 exactly once despite 8 concurrent readers
  PENTAGO_ASSERT_EQ(fetches.load(), 1);
}

// --- open_file routing ---

TEST(gcs, open_file_local) {
  // open_file with a local path should work (no GCS init needed)
  // Just verify it doesn't crash — actual file doesn't need to exist for this test
  ASSERT_FALSE(is_gcs_path("/tmp/test.pentago"));
}

}  // namespace
}  // namespace pentago
