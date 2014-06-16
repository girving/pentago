// Atomic thread safe counters
#pragma once

#include <geode/python/config.h>
namespace pentago {

class counter_t {
  int count;
public:

  counter_t(int count)
    : count(count) {} 

  int operator--() {
    int r = fetch_and_add(&count,-1)-1;
    assert(r>=0);
    return r;
  }

  explicit operator bool() const {
    return count!=0;
  }
};

}
