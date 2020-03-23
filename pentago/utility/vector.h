// Fixed size vectors
#pragma once

#include "pentago/utility/wasm.h"
#include <algorithm>
#include <cassert>
#ifndef __wasm__
#include <boost/functional/hash.hpp>
#endif
NAMESPACE_PENTAGO

#ifndef __wasm__
using std::ostream;
#endif
template<class T, int d> class Vector;

template<class TV> struct scalar_type_helper {
  static_assert(std::is_fundamental<TV>::value);
  typedef TV type;
};
template<class T,int d> struct scalar_type_helper<Vector<T,d>> { typedef T type; };
template<class TV> using scalar_type = typename scalar_type_helper<TV>::type;

template<class T, int d> struct VectorBase {
  T data_[d];

  VectorBase() { std::fill(data_, data_+d, T()); }
  explicit VectorBase(const T& x0) : data_{x0} { static_assert(d == 1); }
  VectorBase(const T& x0, const T& x1) : data_{x0, x1} { static_assert(d == 2); }
  VectorBase(const T& x0, const T& x1, const T& x2) : data_{x0, x1, x2} { static_assert(d == 3); }
  VectorBase(const T& x0, const T& x1, const T& x2, const T& x3)
    : data_{x0, x1, x2, x3} { static_assert(d == 4); }
  VectorBase(const T& x0, const T& x1, const T& x2, const T& x3, const T& x4)
    : data_{x0, x1, x2, x3, x4} { static_assert(d == 5); }
  VectorBase(const T& x0, const T& x1, const T& x2, const T& x3, const T& x4, const T& x5)
    : data_{x0, x1, x2, x3, x4, x5} { static_assert(d == 6); }
  VectorBase(const T& x0, const T& x1, const T& x2, const T& x3, const T& x4, const T& x5, const T& x6)
    : data_{x0, x1, x2, x3, x4, x5, x6} { static_assert(d == 7); }

  T* data() { return data_; }
  const T* data() const { return data_; }
};
template<class T> struct VectorBase<T,0> {
  T* data() { return nullptr; }
  const T* data() const { return nullptr; }
};

template<class T, int d> class Vector : public VectorBase<T,d> {
public:
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;
  constexpr static int m = d;

  using VectorBase<T,d>::data;
  using VectorBase<T,d>::VectorBase;

  template<class S> explicit Vector(const Vector<S,d>& x) {
    for (int i = 0; i < d; i++) data()[i] = x[i];
  }

  constexpr int size() const { return m; }
  constexpr bool empty() const {return m > 0; }

  const T& operator[](const int i) const {
    assert(unsigned(i) < d);
    return data()[i];
  }

  T& operator[](const int i) {
    assert(unsigned(i)<d);
    return data()[i];
  }

  T* begin() { return data(); }
  const T* begin() const { return data(); }
  T* end() { return data() + d; }
  const T* end() const { return data() + d; }
  bool operator==(const Vector& v) const { return std::equal(data(), data() + d, v.data()); }
  bool operator!=(const Vector& v) const { return !(*this == v); }

#define OP(op) \
  Vector& operator op##=(const T& v)      { for (int i=0;i<d;i++) data()[i] op##= v; return *this; } \
  Vector& operator op##=(const Vector& v) { for (int i=0;i<d;i++) data()[i] op##= v[i]; return *this; } \
  Vector operator op(const T& v) const      { auto r = *this; r op##= v; return r; } \
  Vector operator op(const Vector& v) const { auto r = *this; r op##= v; return r; }
OP(+)
OP(-)
OP(*)
OP(/)
OP(|)
OP(&)
OP(^)
#undef OP

  T min() const {
    static_assert(m > 0);
    T r = data()[0];
    for (int i = 1; i < d; i++) r = std::min(r, data()[i]);
    return r;
  }

  T max() const {
    static_assert(m > 0);
    T r = data()[0];
    for (int i = 1; i < d; i++) r = std::max(r, data()[i]);
    return r;
  }

  int argmin() const {
    static_assert(m > 0);
    int r = 0;
    for (int i = 1; i < d; i++)
      if (data()[r] > data()[i])
        r = i;
    return r;
  }

  int argmax() const {
    static_assert(m > 0);
    int r = 0;
    for (int i = 1; i < d; i++)
      if (data()[r] < data()[i])
        r = i;
    return r;
  }

  T sum() const {
    if (!m) return T();
    T r = data()[0];
    for (int i = 1; i < d; i++) r += data()[i];
    return r;
  }

  T product() const {
    if (!m) return 1;
    T r = data()[0];
    for (int i = 1; i < d; i++) r *= data()[i];
    return r;
  }

  int find(const T& x) const {
    for (int i = 0; i < d; i++) if (data()[i] == x) return i;
    return -1;
  }

  bool contains(const T& x) const {
    for (int i = 0; i < d; i++) if (data()[i] == x) return true;
    return false;
  }

  Vector<T,d> sorted() const {
    static_assert(d <= 4);
    Vector<T,d> s = *this;
    switch (d) {
      #define CMP(i, j) if (s[j] < s[i]) std::swap(s[i], s[j]);
      case 0: case 1: break;
      case 2: CMP(0,1) break;
      case 3: CMP(0,1) CMP(1,2) CMP(0,1) break;
      case 4: CMP(0,1) CMP(2,3) CMP(0,2) CMP(1,3) CMP(1,2) break;
      #undef CMP
    }
    return s;
  }

  Vector<T,d+1> insert(const T& x, int i) const {
    Vector<T,d+1> r;
    r[i] = x;
    for (int j = 0; j < i; j++) r[j] = data()[j];
    for (int j = i; j < d; j++) r[j+1] = data()[j];
    return r;
  }

  Vector<T,d-1> remove_index(int i) const {
    Vector<T,d-1> r;
    for (int j = 0; j < i; j++) r[j] = data()[j];
    for (int j = i; j < d-1; j++) r[j] = data()[j+1];
    return r;
  }

  template<int s> Vector<T,s> subset(const Vector<int,s>& I) const {
    Vector<T,s> r;
    for (int i = 0; i < s; i++) r[i] = (*this)[I[i]];
    return r;
  }
};

template<class T,class... Args> static inline auto vec(const Args&... args)
  -> Vector<T,sizeof...(Args)> {
  return Vector<T,sizeof...(Args)>(args...);
}

template<class... Args> static inline auto vec(const Args&... args) {
  return Vector<typename std::common_type_t<Args...>,sizeof...(Args)>(args...);
}

template<class T> static inline T clamp(const T x, const T min, const T max) {
  return x <= min ? min : x >= max ? max : x;
}

template<class T,int d> static inline Vector<T,d>
clamp(const Vector<T,d>& x, const Vector<T,d>& min, const Vector<T,d>& max) {
  Vector<T,d> y;
  for (int i = 0; i < d; i++) y[i] = clamp(x[i], min[i], max[i]);
  return y;
}

template<class T> static inline T cwise_min(const T x, const T y) { return std::min(x, y); }
template<class T> static inline T cwise_max(const T x, const T y) { return std::max(x, y); }

template<class T,int d> static inline Vector<T,d>
cwise_min(const Vector<T,d>& x, const Vector<T,d>& y) {
  Vector<T,d> r;
  for (int i = 0; i < d; i++) r[i] = std::min(x[i], y[i]);
  return r;
}

template<class T,int d> static inline Vector<T,d>
cwise_max(const Vector<T,d>& x, const Vector<T,d>& y) {
  Vector<T,d> r;
  for (int i = 0; i < d; i++) r[i] = std::max(x[i], y[i]);
  return r;
}

template<class T,int d> static inline Vector<T,d>
operator*(const T s, const Vector<T,d>& x) {
  return x * s;
}

template<class T,int a,int b> static inline Vector<T,a+b>
concat(const Vector<T,a>& x, const Vector<T,b>& y) {
  Vector<T,a+b> xy;
  for (int i = 0; i < a; i++) xy[i] = x[i];
  for (int i = 0; i < b; i++) xy[a+i] = y[i];
  return xy;
}

template<class T,int d> static inline T dot(const Vector<T,d>& x, const Vector<T,d>& y) {
  T r = 0;
  for (int i = 0; i < d; i++) r += x[i] * y[i];
  return r;
}

#ifndef __wasm__
template<class T,int d> ostream& operator<<(ostream& out, const Vector<T,d>& v) {
  out << '[';
  for (int i = 0; i < d; i++) {
    if (i) out << ',';
    out << v[i];
  }
  out << ']';
  return out;
}

template<class T,int d> static inline size_t hash_value(const Vector<T,d>& v) {
  size_t h = 0;
  for (const auto& x : v) boost::hash_combine(h, x);
  return h;
}
#endif  // !__wasm__

template<int i,class T,int d> static inline T& get(Vector<T,d>& v) {
  static_assert(0 <= i && i < d);
  return v.data()[i];
}

template<int i,class T,int d> static inline const T& get(const Vector<T,d>& v) {
  static_assert(0 <= i && i < d);
  return v.data()[i];
}

END_NAMESPACE_PENTAGO
namespace std {
#ifndef __wasm__
template<class T,int d> struct hash<pentago::Vector<T,d>> {
  size_t operator()(const pentago::Vector<T,d>& v) const {
    return hash_value(v);
  }
};
#endif  // !__wasm__
template<class T,int d> class tuple_size<PENTAGO_NAMESPACE::Vector<T,d>> {
 public:
  constexpr static int value = d;
};
template<size_t i,class T,int d> class tuple_element<i,PENTAGO_NAMESPACE::Vector<T,d>> {
 public:
  typedef T type;
};
}  // namespace std
