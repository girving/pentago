// Unit tests for GCS internals (no actual GCS access)

#include "pentago/gcs/gcs.h"
#include "pentago/gcs/internal.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

using gcs_internal::json_string;
using gcs_internal::base64url_encode;

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

}  // namespace
}  // namespace pentago
