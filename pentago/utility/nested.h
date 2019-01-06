// Nested arrays stored flat
#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/range.h"
namespace pentago {

Array<int> nested_array_offsets(RawArray<const int> lengths);

template<class T> class Nested {
public:
  typedef typename Array<T>::value_type value_type;

  Array<const int> offsets;
  Array<T> flat;

  Nested()
    : Nested(RawArray<const int>()) {}

  Nested(RawArray<const int> lengths)
    : offsets(nested_array_offsets(lengths))
    , flat(offsets.back()) {}

  Nested(RawArray<const int> lengths, Uninit)
    : offsets(nested_array_offsets(lengths))
    , flat(offsets.back(), uninit) {}

  int size() const { return offsets.size() - 1; }
  int size(int i) const { return offsets[i+1] - offsets[i]; }
  bool empty() const { return !size(); }
  bool valid(int i) const { return unsigned(i)<unsigned(size()); }
  bool valid(int i, int j) const { return valid(i) && unsigned(j) < unsigned(size(i)); }
  int total_size() const { return offsets.back(); }
  Range<int> range(int i) const { return Range<int>(offsets[i], offsets[i+1]); }

  T& operator()(int i, int j) const {
    int index = offsets[i]+j;
    assert(0<=j && index<=offsets[i+1]);
    return flat[index];
  }

  RawArray<T> operator[](int i) const {
    return flat.slice(offsets[i], offsets[i+1]);
  }
};

template<class T> Nested<T> asnested(const vector<vector<T>>& vs) {
  vector<int> lengths;
  for (const auto& v : vs)
    lengths.push_back(v.size());
  Nested<T> nest(lengths);
  int f = 0;
  for (const auto& v : vs)
    for (const auto& x : v)
      nest.flat[f++] = x;
  return nest;
}

}
