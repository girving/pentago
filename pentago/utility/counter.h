// Atomic thread safe counters
#pragma once

#include <atomic>
#include <cassert>
namespace pentago {

class counter_t {
  std::atomic<int> count;
public:

  explicit counter_t(int count)
    : count(count) {} 

  int operator--() {
    const int r = count.fetch_sub(1) - 1;
    assert(r>=0);
    return r;
  }

  explicit operator bool() const {
    return count;
  }
};

}
