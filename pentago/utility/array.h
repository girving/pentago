// Shareable, multidimensional arrays
#pragma once

#include <memory>
#include <type_traits>
#include "pentago/utility/debug.h"
#include "pentago/utility/index.h"
#include "pentago/utility/range.h"
#include "pentago/utility/vector.h"
#ifndef __wasm__
#include <iostream>
#include <vector>
#endif
namespace pentago {

#ifndef __wasm__
using std::default_delete;
using std::ostream;
using std::shared_ptr;
using std::unique_ptr;
using std::vector;
template<class T, int d=1> class Array;
#endif  // !__wasm__
using std::numeric_limits;
template<class T, int d=1> class RawArray;
template<class T> class NdArray;

// Marker for special uninitialized constructors
struct Uninit {};
constexpr static Uninit uninit = Uninit();

template<class T> static inline T* get_pointer(T* p) { return p; }
#ifndef __wasm__
template<class T> static inline T* get_pointer(const shared_ptr<T>& p) { return p.get(); }
#endif  // !__wasm__

template<class Shape> class ArraySizes {
  struct Unusable {};
 public:
  template<class T> using RawArray = Unusable;
  template<class T> using Array = pentago::NdArray<T>;

  int rank() const { return shape_.size(); }
 protected:
  Shape shape_;
  ArraySizes() = default;
  ArraySizes(const Shape& shape) : shape_(shape) {}

  template<int r> void rank_assert() const { assert(rank() == r); }
  constexpr static int rank_if_static = -1;
};

template<int d> class ArraySizes<Vector<int,d>> {
 public:
  template<class T> using RawArray = pentago::RawArray<T,d>;
#ifndef __wasm__
  template<class T> using Array = pentago::Array<T,d>;
#endif  // !__wasm__

  typedef Vector<int,d> Shape;
  constexpr static int rank() { return d; }
 protected:
  Shape shape_;
  ArraySizes() = default;
  ArraySizes(const Shape& shape) : shape_(shape) {}

  template<int r> void rank_assert() const { static_assert(rank() == r); }
  constexpr static int rank_if_static = d;
};

template<class Shape> size_t safe_product(const Shape& shape) {
  static_assert(std::is_same<typename Shape::value_type, int>::value);
  size_t prod = 1;
  for (const int n : shape) {
    prod *= size_t(n);
    GEODE_ASSERT(0 <= n && prod <= numeric_limits<int>::max());
  }
  return prod;
}

template<class T, class Shape, class Data> class ArrayBase : public ArraySizes<Shape> {
  typedef ArraySizes<Shape> Base;
  struct unusable_t {};
protected:
  using Base::shape_;
public:
  typedef std::remove_const_t<T> value_type;
  static const bool is_const = std::is_const<T>::value;
  typedef T& result_type;

protected:
  Data data_;

  ArrayBase() = default;
  ArrayBase(const Shape& shape, const Data& data) : Base(shape), data_(data) {}
public:

  int size() const { Base::template rank_assert<1>(); return shape_[0]; }
  int total_size() const { return shape_.product(); }
  const Shape& shape() const { return shape_; }
  T* data() const { return get_pointer(data_); }

  T& operator[](std::conditional_t<(Base::rank_if_static<=1),const int,unusable_t> i0) const {
    return (*this)(i0);
  }

  auto operator[](std::conditional_t<(Base::rank_if_static>1),const int,unusable_t> i0) const {
    assert(unsigned(i0) < unsigned(shape_[0]));
    const auto sub = shape_.remove_index(0);
    return RawArray<T,Base::rank()-1>(shape_.remove_index(0), data() + i0*sub.product());
  }

  T& operator()(const int i0) const {
    assert(valid(i0));
    return data()[i0];
  }

  T& operator()(const int i0, const int i1) const {
    assert(valid(i0, i1));
    return data()[i0 * shape_[1] + i1];
  }

  T& operator()(const int i0, const int i1, const int i2) const {
    assert(valid(i0, i1, i2));
    return data()[(i0 * shape_[1] + i1) * shape_[2] + i2];
  }

  T& operator()(const int i0, const int i1, const int i2, const int i3) const {
    assert(valid(i0, i1, i2, i3));
    return data()[((i0 * shape_[1] + i1) * shape_[2] + i2) * shape_[3] + i3];
  }

  bool valid(const int i0) const {
    Base::template rank_assert<1>();
    return unsigned(i0) < unsigned(shape_[0]);
  }

  bool valid(const int i0, const int i1) const {
    Base::template rank_assert<2>();
    return unsigned(i0) < unsigned(shape_[0]) &&
           unsigned(i1) < unsigned(shape_[1]);
  }

  bool valid(const int i0, const int i1, const int i2) const {
    Base::template rank_assert<3>();
    return unsigned(i0) < unsigned(shape_[0]) &&
           unsigned(i1) < unsigned(shape_[1]) &&
           unsigned(i2) < unsigned(shape_[2]);
  }

  bool valid(const int i0, const int i1, const int i2, const int i3) const {
    Base::template rank_assert<4>();
    return unsigned(i0) < unsigned(shape_[0]) &&
           unsigned(i1) < unsigned(shape_[1]) &&
           unsigned(i2) < unsigned(shape_[2]) &&
           unsigned(i3) < unsigned(shape_[3]);
  }

  template<int d> T& operator[](const Vector<int,d>& I) const {
    assert(valid(I));
    return data()[index(shape_, I)];
  }

  template<int d> bool valid(const Vector<int,d>& I) const {
    Base::template rank_assert<d>();
    for (int i = 0; i < d; i++)
      if (unsigned(I[i]) >= unsigned(shape_[i]))
        return false;
    return true;
  }

  void fill(const T& c) const {
    std::fill(data(), data() + total_size(), c);
  }
  void zero() const { fill(T()); }

  template<class A,class S,class D> bool operator==(const ArrayBase<A,S,D>& other) const {
    return shape_ == other.shape() && std::equal(data(), data() + total_size(), other.data());
  }

  template<class S,class D> bool operator!=(const ArrayBase<T,S,D>& other) const {
    return !(*this == other);
  }

  T* begin() const { return data(); }
  T* end() const { return data() + size(); }
  T& front() const { return (*this)[0]; }
  T& back() const { return (*this)[size() - 1]; }

  value_type min() const {
    using std::min;
    const int n = total_size();
    GEODE_ASSERT(n);
    auto result = data()[0];
    for (int i = 1; i < n; i++) result = min(result, data()[i]);
    return result;
  }

  value_type max() const {
    using std::max;
    const int n = total_size();
    GEODE_ASSERT(n);
    auto result = data()[0];
    for (int i = 1; i < n; i++) result = max(result, data()[i]);
    return result;
  }

  value_type sum() const {
    auto sum = value_type();
    const int n = total_size();
    for (int i = 0; i < n; i++) sum += data()[i];
    return sum;
  }

  value_type product() const {
    auto prod = 1;
    const int n = total_size();
    for (int i = 0; i < n; i++) prod *= data()[i];
    return prod;
  }

  RawArray<T> flat() const { return RawArray<T>(total_size(), data()); }

  RawArray<T> slice(int start, int end) const {
    assert(0 <= start && start <= end && end <= size());
    return RawArray<T>(end - start, data() + start);
  }

  RawArray<T> slice(Range<int> range) const {
    return slice(range.lo, range.hi);
  }

  template<int d> RawArray<T,d> reshape(const Vector<int,d>& new_shape) const {
    GEODE_ASSERT(safe_product(new_shape) == size_t(total_size()));
    return RawArray<T,d>(new_shape, data());
  }

  // Return a non-owning array for use in threaded code where reference counting is slow
  auto raw() const { return typename Base::template RawArray<T>(shape_, data()); }

#ifndef __wasm__
  Array<T> slice_own(int start, int end) const {
    assert(0 <= start && start <= end && end <= size());
    return Array<T>(vec(end - start), shared_ptr<T>(data_, data() + start));
  }

  template<int d> Array<T,d> reshape_own(const Vector<int,d>& new_shape) const {
    GEODE_ASSERT(safe_product(new_shape) == size_t(total_size()));
    return Array<T,d>(new_shape, data_);
  }

  typename Base::template Array<value_type> copy() const {
    typename Base::template Array<value_type> copy(Base::shape_, uninit);
    std::copy(data(), data() + total_size(), copy.data());
    return copy;
  }
#endif  // !__wasm__
};

#ifndef __wasm__
template<class T, int d> class Array : public ArrayBase<T,Vector<int,d>,shared_ptr<T>> {
  typedef ArrayBase<T,Vector<int,d>,shared_ptr<T>> Base;
  struct Unusable {};
public:
  using typename Base::value_type;

  Array() = default;
  Array(const Vector<int,d> shape, const shared_ptr<T>& data) : Base(shape, data) {}

  explicit Array(const Vector<int,d> shape)
    : Base(shape, shared_ptr<T>(new T[safe_product(shape)](), default_delete<T[]>())) {}

  // TODO(girving): This should avoid constructors even for user defined types
  Array(const Vector<int,d> shape, Uninit)
    : Base(shape, shared_ptr<T>(new T[safe_product(shape)], default_delete<T[]>())) {}

  explicit Array(int m0) : Array(vec(m0)) {}
  explicit Array(int m0, int m1) : Array(vec(m0, m1)) {}
  explicit Array(int m0, int m1, int m2) : Array(vec(m0, m1, m2)) {}
  explicit Array(int m0, int m1, int m2, int m3) : Array(vec(m0, m1, m2, m3)) {}
  Array(int m0, Uninit) : Array(vec(m0), uninit) {}
  Array(int m0, int m1, Uninit) : Array(vec(m0, m1), uninit) {}
  Array(int m0, int m1, int m2, Uninit) : Array(vec(m0, m1, m2), uninit) {}
  Array(int m0, int m1, int m2, int m3, Uninit) : Array(vec(m0, m1, m2, m3), uninit) {}

  Array(const Array<std::conditional_t<std::is_const<T>::value,
                                       std::remove_const_t<T>, Unusable>,d>& nonconst)
    : Base(nonconst.shape(), nonconst.owner()) {}

  const shared_ptr<T>& owner() const { return Base::data_; }

  void clean_memory() {
    *this = Array();
  }

  Array<T> flat_own() const { return Array<T>(vec(Base::total_size()), Base::data_); }

  const Array<const value_type,d>& const_() const {
    return *reinterpret_cast<const Array<const value_type,d>*>(this);
  }

  const Array<value_type,d>& const_cast_() const {
    return *reinterpret_cast<const Array<value_type,d>*>(this);
  }
};
#endif  // !__wasm__

template<class T, int d> class RawArray : public ArrayBase<T,Vector<int,d>,T*> {
  typedef ArrayBase<T,Vector<int,d>,T*> Base;
  struct Unusable {};
public:
  using typename Base::value_type;

  RawArray() : Base(Vector<int,d>(), 0) {}
  RawArray(int size, T* data) : Base(vec(size), data) {}
  RawArray(Vector<int,d> shape, T* data) : Base(shape, data) {}

#ifndef __wasm__
  RawArray(vector<value_type>& x) : RawArray(CHECK_CAST_INT(uint64_t(x.size())), x.data()) {}
  RawArray(const vector<value_type>& x) : RawArray(CHECK_CAST_INT(uint64_t(x.size())), x.data()) {}

  template<class S> RawArray(const Array<S,d>& array)
    : Base(array.shape(), array.data()) {}
#endif  // !__wasm__

  RawArray(const RawArray<std::conditional_t<std::is_const<T>::value,
                                             std::remove_const_t<T>, Unusable>,d>& nonconst)
    : Base(nonconst.shape(), nonconst.data()) {}

  const RawArray<const value_type,d>& const_() const {
    return *reinterpret_cast<const RawArray<const value_type,d>*>(this);
  }

  const RawArray<value_type,d>& const_cast_() const {
    return *reinterpret_cast<const RawArray<value_type,d>*>(this);
  }

#ifndef __wasm__
  using Base::copy;

  void copy(RawArray<const value_type> other) const {
    GEODE_ASSERT(Base::shape_ == other.shape());
    std::copy(other.data(), other.data() + other.total_size(), Base::data_);
  }
#endif  // !__wasm__
};

#ifndef __wasm__
template<class T> class NdArray : public ArrayBase<T,Array<const int>,shared_ptr<T>> {
  typedef ArrayBase<T,Array<const int>,shared_ptr<T>> Base;
  struct Unusable {};
public:
  NdArray() = default;
  NdArray(const Array<const int>& shape, const shared_ptr<T>& data) : Base(shape, data) {}

  explicit NdArray(const Array<const int>& shape)
    : Base(shape, shared_ptr<T>(new T[safe_product(shape)](), default_delete<T[]>())) {}

  NdArray(const Array<const int>& shape, Uninit)
    : Base(shape, shared_ptr<T>(new T[safe_product(shape)], default_delete<T[]>())) {}

  NdArray(const NdArray<std::conditional_t<std::is_const<T>::value,
                                           std::remove_const_t<T>, Unusable>>& nonconst)
    : Base(nonconst.shape(), nonconst.owner()) {}

  const shared_ptr<T>& owner() const { return Base::data_; }
};

template<class T> static inline Array<T>
concat(RawArray<const T> x, RawArray<const T> y) {
  const int xn = x.size(), yn = y.size();
  Array<T> xy(xn + yn, uninit);
  for (int i = 0; i < xn; i++) xy[i] = x[i];
  for (int i = 0; i < yn; i++) xy[xn+i] = y[i];
  return xy;
}

template<class T,int d> static inline const Array<T,d>& asarray(const Array<T,d>& x) { return x; }
template<class T> static inline RawArray<T> asarray(vector<T>& x) { return x; }
template<class T> static inline RawArray<const T> asarray(const vector<T>& x) { return x; }
#endif  // !__wasm__

template<class T,int d> static inline const RawArray<T,d>& asarray(const RawArray<T,d>& x) { return x; }
template<class T,int d> static inline const RawArray<T> asarray(T (&x)[d]) { return RawArray<T>(d, x); }
template<class T,int d> static inline const RawArray<T> asarray(Vector<T,d>& v) {
  return RawArray<T>(d, v.begin());
}
template<class T,int d> static inline const RawArray<const T> asarray(const Vector<T,d>& v) {
  return RawArray<const T>(d, v.begin());
}

#ifndef __wasm__
template<class T, class A> inline void extend(vector<T>& dst, const A& src) {
  dst.insert(dst.end(), src.begin(), src.end());
}
#endif
template<class C, class T> inline bool contains(const C& container, const T& x) {
  return container.find(x) != container.end();
}
template<class C, class T> inline auto check_get(C& container, const T& x) {
  const auto it = container.find(x);
  GEODE_ASSERT(it != container.end());
  return it->second;
}
template<class C, class T> inline auto get_pointer(C& container, const T& x) {
  const auto it = container.find(x);
  return it == container.end() ? nullptr : &it->second;
}

#ifndef __wasm__
template<class A> void array_stream_helper(ostream& out, const A& a, const int dim, const int offset) {
  if (dim == a.rank()) {
    out << a.data()[offset];
  } else {
    const int n = a.shape()[dim];
    out << '[';
    for (int i = 0; i < n; i++) {
      if (i) out << ',';
      array_stream_helper(out, a, dim+1, offset * n + i);
    }
    out << ']';
  }
}

template<class T,class S,class D> ostream& operator<<(ostream& out, const ArrayBase<T,S,D>& a) {
  array_stream_helper(out, a, 0, 0);
  return out;
}
#endif  // !__wasm__

}
