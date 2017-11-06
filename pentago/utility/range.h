#pragma once

#include <type_traits>
#include <cassert>
namespace pentago {

using std::declval;
using std::enable_if_t;

template<class Iter, class Enable=void> struct Range {
  typedef decltype(*declval<Iter>()) reference;

  Iter lo, hi;

  Range() = default;

  Range(const Iter& lo, const Iter& hi)
    : lo(lo), hi(hi) {}

  const Iter& begin() const { return lo; }
  const Iter& end() const { return hi; }

  bool empty() const { return !(lo != hi); }
  auto size() const -> decltype(declval<Iter>() - declval<Iter>()) { return hi - lo; }
  template<class I> reference operator[](const I i) const { assert(i < hi - lo); return *(lo + i); }

  reference front() const { assert(lo != hi); return *lo; }
  reference back() const { assert(lo != hi); return *(hi - 1); }
};

template<class I> struct Range<I, std::enable_if_t<std::is_integral<I>::value>> {
  I lo, hi;

  struct Iter {
    I i;
    explicit Iter(I i) : i(i) {}
    bool operator!=(Iter j) const { return i != j.i; }
    void operator++() { ++i; }
    void operator--() { --i; }
    I operator*() const { return i; }
  };

  Range()
    : lo(), hi() {}

  Range(const I lo, const I hi)
    : lo(lo), hi(hi) {
    assert(lo <= hi);
  }

  bool empty() const { return hi == lo; }
  I size() const { return hi - lo; }

  Iter begin() const { return Iter(lo); }
  Iter end() const { return Iter(hi); }

  I operator[](const I i) const { assert(0 <= i && i < hi - lo); return lo + i; }

  I front() const { assert(lo < hi); return lo; }
  I back() const { assert(lo < hi); return hi - 1; }

  bool contains(I i) const { return lo <= i && i < hi; }

  bool operator==(const Range r) const { return lo == r.lo && hi == r.hi; }
  bool operator!=(const Range r) const { return lo != r.lo || hi != r.hi; }
};

template<class Iter> static inline Range<Iter> range(const Iter& lo, const Iter& hi) {
  return Range<Iter>(lo,hi);
}

template<class I> static inline Range<I> range(I n) {
  static_assert(std::is_integral<I>::value, "single argument range must take an integral type");
  return Range<I>(0,n);
}

template<class Iter,class I> static inline auto operator+(const Range<Iter>& r, const I n)
  -> enable_if_t<std::is_integral<I>::value, decltype(range(r.lo + n, r.hi + n))> {
  return range(r.lo + n, r.hi + n);
}

template<class Iter,class I> static inline auto operator-(const Range<Iter>& r, const I n)
  -> enable_if_t<std::is_integral<I>::value, decltype(range(r.lo - n, r.hi - n))> {
  return range(r.lo - n, r.hi - n);
}

template<class Iter,class I> static inline auto operator+(const I n, const Range<Iter>& r)
  -> enable_if_t<std::is_integral<I>::value, decltype(range(r.lo + n, r.hi + n))> {
  return range(r.lo + n, r.hi + n);
}

// Partition a loop into chunks based on the total number of threads.  Returns a half open interval.
template<class I,class TI> Range<I>
partition_loop(const I loop_steps, const TI threads, const TI thread);

// Inverse of partition_loop: map an index to the thread that owns it
template<class I,class TI> TI
partition_loop_inverse(const I loop_steps, const TI threads, const I index);

}
