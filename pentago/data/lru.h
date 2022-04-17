// LRU cache
#pragma once

#include "pentago/utility/debug.h"
#include "pentago/utility/noncopyable.h"
#include <list>
#include <unordered_map>
#include <boost/functional/hash.hpp>
namespace pentago {

using std::get;
using std::list;
using std::make_tuple;
using std::tuple;
using std::unordered_map;

template<class K,class V> class lru_t : noncopyable_t {
  // Least to most recently used
  mutable list<tuple<K,V>> order;

  // Key to position in order
  unordered_map<K,typename list<tuple<K,V>>::iterator,boost::hash<K>> table;
public:

  void add(const K key, const V& value) {
    table.insert(make_pair(key,order.insert(order.end(),make_tuple(key,value))));
  }

  const V* get(const K key) const {
    const auto it = table.find(key);
    if (it == table.end())
      return 0;
    order.splice(order.end(), order, it->second); // Move element to back
    return &std::get<1>(*it->second);
  }

  tuple<K,V> drop() {
    GEODE_ASSERT(order.size());
    const auto r = order.front();
    table.erase(std::get<0>(r));
    order.pop_front();
    return r;
  }
};

}
