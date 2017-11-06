#include "pentago/data/lru.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

using std::make_tuple;

TEST(lru, lru) {
  lru_t<int,string> lru;
  lru.add(7,"7");
  lru.add(1,"1");
  ASSERT_EQ(lru.drop(), make_tuple(7,string("7")));
  lru.add(9,"9");
  ASSERT_FALSE(lru.get(2));
  const string* p = lru.get(1);
  ASSERT_TRUE(p && *p=="1");
  ASSERT_EQ(lru.drop(), make_tuple(9,string("9")));
  ASSERT_FALSE(lru.get(9));
}

}  // namespace
}  // namespace pentago
