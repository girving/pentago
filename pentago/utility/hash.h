#pragma once

#include <tuple>
namespace pentago {

template<class T> static inline void hash_combine(size_t& seed, const T& v) {
  // Following boost::hash_combine
  seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

}  // namespace pentago
namespace std {
template<class... Args> struct hash<tuple<Args...>> {
  size_t operator()(const tuple<Args...>& t) const {
    size_t h = 0;
    std::apply([&h](auto&&... args) { (pentago::hash_combine(h, args), ...); }, t);
    return h;
  }
};
}  // namespace std
