// Interface to snappy

#include "pentago/end/fast_compress.h"
#include "pentago/data/filter.h"
#include "pentago/utility/thread.h"
#include "pentago/end/config.h"
#include "pentago/utility/ceil_div.h"
#include "pentago/utility/char_view.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/memory.h"
#include "pentago/utility/array.h"
#include "pentago/utility/sqr.h"
#include "snappy.h"
namespace pentago {
namespace end {

const int raw_max_fast_compressed_size =
    int(1+snappy::MaxCompressedLength(sizeof(Vector<super_t,2>) * sqr(sqr(block_size))));

int fast_compress(RawArray<Vector<super_t,2>> uncompressed, RawArray<uint8_t> compressed,
                  const event_t event) {
  thread_time_t time(snappy_kind,event);
  // Filter
  if (snappy_filter)
    interleave(uncompressed);
  // Compress
  const size_t input_size = sizeof(Vector<super_t,2>)*uncompressed.size();
  const size_t max_output = 1 + snappy::MaxCompressedLength(input_size);
  GEODE_ASSERT(size_t(compressed.size()) >= max_output,
               format("fast_compress: compressed.size = %d < 1 + snappy_max_output(%d) = %d",
                      compressed.size(), input_size, max_output));
  size_t output_size;
  compressed[0] = 1; // Set compressed flag
  snappy::RawCompress((const char*)uncompressed.data(),input_size,(char*)compressed.data()+1,&output_size);
  if (output_size>=input_size) {
    compressed[0] = 0; // Fall back to uncompressed mode
    output_size = input_size;
    memcpy(compressed.data()+1,uncompressed.data(),input_size);
  }
  return int(output_size+1);
}

// Same as fast_uncompress, but doesn't require an exact size match in the uncompressed buffer and returns the correct size.
static inline int fast_uncompress_helper(RawArray<const uint8_t> compressed, RawArray<Vector<super_t,2>> uncompressed, const event_t event) {
  const auto n = sizeof(Vector<super_t,2>);
  thread_time_t time(unsnappy_kind,event);
  // Uncompress
  GEODE_ASSERT(compressed.size());
  int count;
  if (compressed[0]) { // Compressed mode
    size_t uncompressed_size;
    GEODE_ASSERT(snappy::GetUncompressedLength((const char*)compressed.data()+1,compressed.size()-1,&uncompressed_size));
    if (uncompressed_size > memory_usage(uncompressed))
      die("fast_uncompress: expected size at most %zu, got %zu, event 0x%llx",(size_t)uncompressed.size(),uncompressed_size,event);
    if (uncompressed_size & (n-1))
      die("fast_uncompress: expected size a multiple of %zu, got %zu, event 0x%llx",n,uncompressed_size,event);
    GEODE_ASSERT(snappy::RawUncompress((const char*)compressed.data()+1,compressed.size()-1,(char*)uncompressed.data()));
    count = int(uncompressed_size/n);
  } else {
    GEODE_ASSERT((size_t)compressed.size()<=1+memory_usage(uncompressed) && !((compressed.size()-1)&(n-1)));
    memcpy(uncompressed.data(),compressed.data()+1,compressed.size()-1);
    count = compressed.size()/n;
  }
  // Unfilter
  if (snappy_filter)
    uninterleave(uncompressed.slice(0,count));
  return count;
}

void fast_uncompress(RawArray<const uint8_t> compressed, RawArray<Vector<super_t,2>> uncompressed,
                     const event_t event) {
  const int count = fast_uncompress_helper(compressed,uncompressed,event);
  if (count != uncompressed.size())
    die("fast_uncompress: expected count %d, got %d, event 0x%llx",uncompressed.size(),count,event);
}

#ifdef __clang__
// __thread is broken under clang on BlueGene, so use pthread function calls instead
#define USE_PTHREADS 1
#else
#define USE_PTHREADS 0
#endif

#if USE_PTHREADS
static pthread_key_t buffer_key;
__attribute__((constructor)) static void create_buffer_key() {
  const int r = pthread_key_create(&buffer_key,free);
  if (r)
    die("local_fast_compress/uncompress: failed to create thread local buffer pthread key: %s",strerror(r));
}
#endif

// Thread local temporary buffer for local compression and decompression.
static inline RawArray<Vector<super_t,2>> local_buffer() {
  const int count = ceil_div(raw_max_fast_compressed_size, int(sizeof(Vector<super_t,2>)));
#if USE_PTHREADS
  auto buffer = (Vector<super_t,2>*)pthread_getspecific(buffer_key);
#else
  static __thread Vector<super_t,2>* buffer = 0;
#endif
  if (!buffer) {
    buffer = (Vector<super_t,2>*)malloc(sizeof(Vector<super_t,2>)*count);
    if (!buffer)
      die("local_fast_compress/uncompress: failed to allocate thread local buffer of size %zu",sizeof(Vector<super_t,2>)*count);
#if USE_PTHREADS
    pthread_setspecific(buffer_key,buffer); 
#endif
  }
  return RawArray<Vector<super_t,2>>(count,buffer);
}

RawArray<uint8_t> local_fast_compress(RawArray<Vector<super_t,2>> uncompressed, const event_t event) {
  const auto compressed = char_view(local_buffer());
  return compressed.slice(0, fast_compress(uncompressed,compressed,event));
}
   
RawArray<Vector<super_t,2>> local_fast_uncompress(RawArray<const uint8_t> compressed,
                                                  const event_t event) {
  const auto uncompressed = local_buffer();
  return uncompressed.slice(0,fast_uncompress_helper(compressed,uncompressed,event));
}

}
}
