// Integer-safe gtest assertions that handle mixed signed/unsigned correctly
#pragma once

#include <type_traits>
#include <utility>
#include "gtest/gtest.h"

namespace pentago {

template<class T> constexpr bool is_safe_integer_v =
    std::is_integral_v<T> && !std::is_same_v<T, bool>;

template<class A, class B> bool safe_cmp_eq(const A& a, const B& b) {
  if constexpr (is_safe_integer_v<A> && is_safe_integer_v<B>) return std::cmp_equal(a, b);
  else return a == b;
}
template<class A, class B> bool safe_cmp_lt(const A& a, const B& b) {
  if constexpr (is_safe_integer_v<A> && is_safe_integer_v<B>) return std::cmp_less(a, b);
  else return a < b;
}
template<class A, class B> bool safe_cmp_gt(const A& a, const B& b) {
  if constexpr (is_safe_integer_v<A> && is_safe_integer_v<B>) return std::cmp_greater(a, b);
  else return a > b;
}
template<class A, class B> bool safe_cmp_le(const A& a, const B& b) {
  if constexpr (is_safe_integer_v<A> && is_safe_integer_v<B>) return std::cmp_less_equal(a, b);
  else return a <= b;
}
template<class A, class B> bool safe_cmp_ge(const A& a, const B& b) {
  if constexpr (is_safe_integer_v<A> && is_safe_integer_v<B>) return std::cmp_greater_equal(a, b);
  else return a >= b;
}

template<class A, class B>
testing::AssertionResult SafeEq(const char* as, const char* bs, const A& a, const B& b) {
  if (safe_cmp_eq(a, b)) return testing::AssertionSuccess();
  return testing::AssertionFailure() << as << " (" << a << ") != " << bs << " (" << b << ")";
}
template<class A, class B>
testing::AssertionResult SafeLt(const char* as, const char* bs, const A& a, const B& b) {
  if (safe_cmp_lt(a, b)) return testing::AssertionSuccess();
  return testing::AssertionFailure() << as << " (" << a << ") >= " << bs << " (" << b << ")";
}
template<class A, class B>
testing::AssertionResult SafeGt(const char* as, const char* bs, const A& a, const B& b) {
  if (safe_cmp_gt(a, b)) return testing::AssertionSuccess();
  return testing::AssertionFailure() << as << " (" << a << ") <= " << bs << " (" << b << ")";
}
template<class A, class B>
testing::AssertionResult SafeLe(const char* as, const char* bs, const A& a, const B& b) {
  if (safe_cmp_le(a, b)) return testing::AssertionSuccess();
  return testing::AssertionFailure() << as << " (" << a << ") > " << bs << " (" << b << ")";
}
template<class A, class B>
testing::AssertionResult SafeGe(const char* as, const char* bs, const A& a, const B& b) {
  if (safe_cmp_ge(a, b)) return testing::AssertionSuccess();
  return testing::AssertionFailure() << as << " (" << a << ") < " << bs << " (" << b << ")";
}

}  // namespace pentago

#define PENTAGO_ASSERT_EQ(a, b) ({ ASSERT_PRED_FORMAT2(pentago::SafeEq, a, b); })
#define PENTAGO_ASSERT_LT(a, b) ({ ASSERT_PRED_FORMAT2(pentago::SafeLt, a, b); })
#define PENTAGO_ASSERT_GT(a, b) ({ ASSERT_PRED_FORMAT2(pentago::SafeGt, a, b); })
#define PENTAGO_ASSERT_LE(a, b) ({ ASSERT_PRED_FORMAT2(pentago::SafeLe, a, b); })
#define PENTAGO_ASSERT_GE(a, b) ({ ASSERT_PRED_FORMAT2(pentago::SafeGe, a, b); })
