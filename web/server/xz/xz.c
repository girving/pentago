// xz (LZMA2) decompression for WebAssembly
//
// A thin wrapper around liblzma's single-call stream decoder, compiled freestanding (no libc)
// by ./build-wasm.  Memory is a bump arena over wasm linear memory: the caller reserves input
// and output buffers with xz_alloc, calls xz_decompress, copies the result out, and xz_reset
// rewinds everything.  liblzma's own allocations (dictionary, decoder state) land in the same
// arena via the malloc/calloc/free below, so nothing is ever individually freed.

#include <lzma.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// Returned by xz_decompress when input remains after the end of the stream
#define XZ_TRAILING_DATA 100

// Bump allocator above the linker-provided heap base
extern unsigned char __heap_base;
static unsigned char* heap = &__heap_base;

static void* arena_alloc(const size_t size) {
  const uintptr_t align = 8;
  unsigned char* begin = (unsigned char*)(((uintptr_t)heap + align - 1) & ~(align - 1));
  if (size > UINTPTR_MAX - (uintptr_t)begin)
    return NULL;
  unsigned char* end = begin + size;
  const uintptr_t have = (uintptr_t)__builtin_wasm_memory_size(0) << 16;
  if ((uintptr_t)end > have) {
    const uintptr_t pages = ((uintptr_t)end - have + 65535) >> 16;
    if (__builtin_wasm_memory_grow(0, pages) < 0)
      return NULL;
  }
  heap = end;
  return begin;
}

// libc subset used by liblzma's decoders.  -ffreestanding implies -fno-builtin, so clang
// won't turn these loops back into calls to themselves.
void* malloc(const size_t size) { return arena_alloc(size); }
void* calloc(const size_t n, const size_t size) {
  if (size && n > SIZE_MAX / size)
    return NULL;
  void* p = arena_alloc(n * size);
  return p ? memset(p, 0, n * size) : NULL;
}
void free(void* p) { (void)p; }  // Reclaimed wholesale by xz_reset
void* memcpy(void* dst, const void* src, size_t n) {
  unsigned char* d = dst;
  const unsigned char* s = src;
  while (n--) *d++ = *s++;
  return dst;
}
void* memmove(void* dst, const void* src, size_t n) {
  unsigned char* d = dst;
  const unsigned char* s = src;
  if (d < s) {
    while (n--) *d++ = *s++;
  } else {
    d += n; s += n;
    while (n--) *--d = *--s;
  }
  return dst;
}
void* memset(void* s, const int c, size_t n) {
  unsigned char* p = s;
  while (n--) *p++ = (unsigned char)c;
  return s;
}
int memcmp(const void* a, const void* b, size_t n) {
  const unsigned char* x = a;
  const unsigned char* y = b;
  for (; n--; x++, y++)
    if (*x != *y)
      return *x - *y;
  return 0;
}

// Exported interface

// Reserve size bytes for the caller (input or output buffers); NULL if memory can't grow
void* xz_alloc(const size_t size) { return arena_alloc(size); }

// Release everything allocated since startup
void xz_reset(void) { heap = &__heap_base; }

// Decompress one complete .xz stream, in[in_size], into out[out_size].  Returns the number of
// bytes produced, or a negative lzma_ret (or -XZ_TRAILING_DATA) on failure.  As in
// pentago/data/compress.cc, the integrity check (CRC64 for our data) is verified and streams
// without one are rejected.  A memory limit well above our preset 6 streams' 8 MiB dictionary
// bounds what a corrupt header can make us allocate.
int32_t xz_decompress(const uint8_t* in, const size_t in_size, uint8_t* out, const size_t out_size) {
  uint64_t memlimit = (uint64_t)128 << 20;
  size_t in_pos = 0, out_pos = 0;
  const lzma_ret r = lzma_stream_buffer_decode(&memlimit, LZMA_TELL_NO_CHECK | LZMA_TELL_UNSUPPORTED_CHECK, NULL,
                                               in, &in_pos, in_size, out, &out_pos, out_size);
  if (r != LZMA_OK)
    return -(int32_t)r;
  if (in_pos != in_size)
    return -XZ_TRAILING_DATA;
  return (int32_t)out_pos;
}
