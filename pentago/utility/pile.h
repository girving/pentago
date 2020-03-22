// Stack-friendly appendable storage
#pragma once

namespace pentago {

template<class T,int limit_> class pile {
  static_assert(std::is_trivially_destructible_v<T>);
  typedef std::aligned_storage_t<sizeof(T),alignof(T)> uninit_t;
  static_assert(std::is_trivially_constructible_v<uninit_t>);

  int size_ = 0;
  uninit_t data_[limit_];

  T* data() { return reinterpret_cast<T*>(data_); }
  const T* data() const { return reinterpret_cast<const T*>(data_); }
public:
  static const int limit = limit_;

  pile() = default;

  void append(const T& x) {
    assert(size_ < limit);
    data()[size_++] = x;
  }

  void clear() { size_ = 0; }

  int size() const { return size_; }
  const T& operator[](const int i) const { GEODE_ASSERT(unsigned(i) < unsigned(size_)); return data()[i]; }
  const T* begin() const { return data(); }
  const T* end() const { return data() + size_; }
};

}  // namespace pentago
