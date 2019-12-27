// Stack-friendly appendable storage
#pragma once

namespace pentago {

template<class T,int limit_> class pile {
public:
  static const int limit = limit_;
private:
  int size_ = 0;
  T data[limit];
public:
  pile() = default;

  void append(const T& x) {
    GEODE_ASSERT(size_ < limit);
    data[size_++] = x;
  }

  int size() const { return size_; }
  const T& operator[](const int i) const { GEODE_ASSERT(unsigned(i) < unsigned(size_)); return data[i]; }
  const T* begin() const { return data; }
  const T* end() const { return data + size_; }
};

}  // namespace pentago
