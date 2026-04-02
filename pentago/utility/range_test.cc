// Range tests

#include "pentago/utility/range.h"
#include "pentago/utility/exceptions.h"
#include "pentago/utility/test_assert.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

TEST(range, parse_both) {
  const auto r = parse_range("3:7", 10);
  PENTAGO_ASSERT_EQ(r.lo, 3);
  PENTAGO_ASSERT_EQ(r.hi, 7);
}

TEST(range, parse_lo_only) {
  const auto r = parse_range("5:", 10);
  PENTAGO_ASSERT_EQ(r.lo, 5);
  PENTAGO_ASSERT_EQ(r.hi, 10);
}

TEST(range, parse_hi_only) {
  const auto r = parse_range(":5", 10);
  PENTAGO_ASSERT_EQ(r.lo, 0);
  PENTAGO_ASSERT_EQ(r.hi, 5);
}

TEST(range, parse_empty) {
  const auto r = parse_range(":", 10);
  PENTAGO_ASSERT_EQ(r.lo, 0);
  PENTAGO_ASSERT_EQ(r.hi, 10);
}

TEST(range, parse_zero_total) {
  const auto r = parse_range(":", 0);
  PENTAGO_ASSERT_EQ(r.lo, 0);
  PENTAGO_ASSERT_EQ(r.hi, 0);
}

TEST(range, parse_equal) {
  const auto r = parse_range("5:5", 10);
  PENTAGO_ASSERT_EQ(r.lo, 5);
  PENTAGO_ASSERT_EQ(r.hi, 5);
  ASSERT_TRUE(r.empty());
}

TEST(range, parse_full) {
  const auto r = parse_range("0:10", 10);
  PENTAGO_ASSERT_EQ(r.lo, 0);
  PENTAGO_ASSERT_EQ(r.hi, 10);
}

TEST(range, parse_invalid_no_colon) {
  ASSERT_THROW(parse_range("5", 10), RuntimeError);
}

TEST(range, parse_invalid_backwards) {
  ASSERT_THROW(parse_range("7:3", 10), RuntimeError);
}

TEST(range, parse_invalid_negative) {
  ASSERT_THROW(parse_range("-1:5", 10), RuntimeError);
}

TEST(range, parse_invalid_overflow) {
  ASSERT_THROW(parse_range("0:11", 10), RuntimeError);
}

TEST(range, parse_invalid_nonnumeric) {
  ASSERT_THROW(parse_range("abc:5", 10), ValueError);
}

}  // namespace
}  // namespace pentago
