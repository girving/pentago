// Sparse storage of a bunch of arrays taking advantage of virtual memory

#include <pentago/end/sparse_store.h>
#include <pentago/end/fast_compress.h>
#include <pentago/utility/ceil_div.h>
#include <pentago/utility/debug.h>
#include <pentago/utility/memory.h>
#include <sys/mman.h>
#include <errno.h>
namespace pentago {
namespace end {

static const int page_size = getpagesize();

sparse_store_t::sparse_store_t(const int count, const int max_size)
  : sizes(count)
  , max_array_size((max_size+page_size-1)&~(page_size-1))
  , sparse_size((size_t)count*max_array_size)
  , sparse_base(sparse_size?(uint8_t*)mmap(0,sparse_size,PROT_READ|PROT_WRITE,MAP_ANON|MAP_PRIVATE,-1,0):0) {
  if (sparse_base==MAP_FAILED)
    THROW(RuntimeError,"anonymous mmap of size %zu failed, %s",sparse_size,strerror(errno));
  memset(sizes.data(),0,memory_usage(sizes));
}

sparse_store_t::~sparse_store_t() {
  if (sparse_base) {
    munmap(sparse_base,sparse_size);
    size_t total = 0;
    for (const auto& info : sizes)
      total += info.size;
    report_large_alloc(-total);
  }
}

uint64_t sparse_store_t::current_memory_usage() const {
  uint64_t total = memory_usage(sizes);
  for (auto& info : sizes)
    total += (info.peak_size+page_size-1)&~(page_size-1);
  return total;
}

uint64_t sparse_store_t::estimate_peak_memory_usage(int count, uint64_t total_size) {
  return count*(sizeof(sizes_t)+page_size/2)+total_size;
}

void sparse_store_t::set_size(int array, int size) const {
  auto& info = sizes[array];
  report_large_alloc(size-info.size);
  info.size = size;
  info.peak_size = max(info.peak_size,size);
}

RawArray<uint8_t> sparse_store_t::whole_buffer(int array) const {
  return RawArray<uint8_t>(max_array_size,sparse_base+(size_t)max_array_size*array);
}

RawArray<uint8_t> sparse_store_t::current_buffer(int array) const {
  return RawArray<uint8_t>(sizes[array].size,sparse_base+(size_t)max_array_size*array);
}

void sparse_store_t::compress_and_set(int array, RawArray<Vector<super_t,2>> uncompressed, event_t event) {
  set_size(array,fast_compress(uncompressed,whole_buffer(array),event));
}

}
}
