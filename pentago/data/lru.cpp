// LRU cache

#include <pentago/data/lru.h>
#include <geode/python/wrap.h>
namespace pentago {

static void lru_test() {
  lru_t<int,string> lru;
  lru.add(7,"7");
  lru.add(1,"1");
  GEODE_ASSERT(lru.drop()==tuple(7,string("7")));
  lru.add(9,"9");
  GEODE_ASSERT(!lru.get(2));
  const string* p = lru.get(1);
  GEODE_ASSERT(p && *p=="1");
  GEODE_ASSERT(lru.drop()==tuple(9,string("9")));
  GEODE_ASSERT(!lru.get(9));
}

}
using namespace pentago;

void wrap_lru() {
  GEODE_FUNCTION(lru_test)
}
