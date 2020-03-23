// WebAssembly allocation

#ifdef __wasm__
#include "pentago/utility/wasm_alloc.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/debug.h"
namespace pentago {

static const size_t page_size = 65536;
extern "C" unsigned char __heap_base;
static size_t next = reinterpret_cast<size_t>(&__heap_base);

void* malloc(size_t size) {
  // Ensure alignment
  const size_t align = 8;
  const size_t mask = ~(align - 1);
  next = (next + align - 1) & mask;
  const size_t aligned_size = (size + align - 1) & mask;

  // Plan where our allocation will go
  const auto begin = reinterpret_cast<void*>(next);
  next += aligned_size;

  // Ensure sufficient space.  We want
  //   next <= page_size * (heap_pages + delta)
  //   next - page_size * heap_pages <= page_size * delta
  const int heap_pages = __builtin_wasm_memory_size(0);
  if (next > page_size * heap_pages) {
    const int delta = ceil_div(next - page_size * heap_pages, page_size);
    const int r = __builtin_wasm_memory_grow(0, delta);
    if (r < 0) die("malloc failed");
  }

  // All done!
  return begin;
}

}  // namespace pentago
#endif  // __wasm__
