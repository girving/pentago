// Noncopyable
#pragma once

namespace pentago {

struct noncopyable_t {
  noncopyable_t() = default;
  noncopyable_t(const noncopyable_t&) = delete;
  void operator=(const noncopyable_t&) = delete;
};

}  // namespace pentago
