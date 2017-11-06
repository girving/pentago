// Insertion sort
//
// Intended for high performance situations where all key and value
// arrays are stack allocated, and therefore not contained in any
// fancy data structures.
#pragma once

#include <algorithm>
namespace pentago {

using std::tuple;
using std::tie;

#define PASS(expression) { const int _pass_helper[] __attribute__((unused)) = {((expression),1)...}; }

// Insertion sort values by keys
template<class Key,class... Values> static inline void
insertion_sort(const int n, Key* keys, Values*... values) {
  for (int i=1;i<n;i++) {
    const Key key = keys[i];
    const tuple<Values...> value(values[i]...);
    int j = i-1;
    while (j>=0 && keys[j]>key) {
      keys[j+1] = keys[j];
      PASS(values[j+1] = values[j]);
      j--;
    }
    keys[j+1] = key;
    tie(values[j+1]...) = value;
  }
}

#undef PASS

struct LexLess {
  template<class A> bool operator()(const A& x, const A& y) const {
    return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
  };
};

}
