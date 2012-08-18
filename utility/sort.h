// Insertion sort
#pragma once

#include <other/core/structure/Tuple.h>
#include <other/core/utility/pass.h>
namespace pentago {

using namespace other;

// Insertion sort values by keys
template<class Key,class... Values> static inline void insertion_sort(const int n, Key* keys, Values*... values) {
  for (int i=1;i<n;i++) {
    const Key key = keys[i];
    const Tuple<Values...> value(values[i]...);
    int j = i-1;
    while (j>=0 && keys[j]>key) {
      keys[j+1] = keys[j];
      OTHER_PASS(values[j+1] = values[j]);
      j--;
    }
    keys[j+1] = key;
    value.get(values[j+1]...);
  }
}

}
