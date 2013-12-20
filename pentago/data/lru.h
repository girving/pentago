// LRU cache
#pragma once

#include <pentago/utility/debug.h>
#include <geode/structure/Tuple.h>
#include <geode/utility/Hasher.h>
#include <geode/utility/tr1.h>
#include <list>
namespace pentago {

using std::list;

template<class K,class V> class lru_t {
  mutable list<Tuple<K,V>> order; // least to most recently used
  unordered_map<K,typename list<Tuple<K,V>>::iterator,Hasher> table; // key to position in order
public:

  void add(const K key, const V& value) {
    table.insert(make_pair(key,order.insert(order.end(),tuple(key,value))));
  }

  const V* get(const K key) const {
    const auto it = table.find(key);
    if (it == table.end())
      return 0;
    order.splice(order.end(),order,it->second); // Move element to back
    return &it->second->y;
  }

  Tuple<K,V> drop() {
    GEODE_ASSERT(order.size());
    const auto r = order.front();
    table.erase(r.x);
    order.pop_front();
    return r;
  }
};

}
