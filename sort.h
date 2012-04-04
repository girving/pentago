// Insertion sort

#pragma once

namespace pentago {

// Insertion sort values by keys
template<class T,class K> static inline void insertion_sort(T* values, K* keys, const int n) {
  for (int i=1;i<n;i++) {
    const T value = values[i];
    const K key = keys[i];
    int j = i-1;
    while (j>=0 && keys[j]>key) {
      values[j+1] = values[j];
      keys[j+1] = keys[j];
      j--;
    }
    values[j+1] = value;
    keys[j+1] = key;
  }
}

}
