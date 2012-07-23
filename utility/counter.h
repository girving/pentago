// Thread safe counters
#pragma once

#include <other/core/python/config.h>
#include <other/core/utility/safe_bool.h>
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

  operator SafeBool() const {
    return safe_bool(count!=0);
  }
};

}
