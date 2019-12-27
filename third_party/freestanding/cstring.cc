// Pentago WebAssembly definitions of necessary standard library functions

// If we get here (i.e., inlining didn't happen), use our own slow versions

#include <cstddef>

extern "C" void* memcpy(void* dst, const void* src, size_t n) {
  auto d = reinterpret_cast<char*>(dst);
  auto s = reinterpret_cast<const char*>(src);
  while (n--) *d++ = *s++;
  return dst;
}

extern "C" void* memmove(void* dst, const void* src, size_t n) {
  // Warning: assumes < works correctly with pointers
  // (see https://quuxplusone.github.io/blog/2019/01/20/std-less-nightmare)
  if (dst < src)
    memcpy(dst, src, n);
  else {
    auto d = reinterpret_cast<char*>(dst) + n;
    auto s = reinterpret_cast<const char*>(src) + n;
    while (n--) *--d = *--s;
  }
  return dst;
}

extern "C" void* memset(void* s, int c, size_t n) {
  auto p = reinterpret_cast<char*>(s);
  while (n--) *p++ = c;
  return s;
}

