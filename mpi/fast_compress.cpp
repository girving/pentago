// Interface to snappy

#include <pentago/mpi/fast_compress.h>
#include <pentago/mpi/config.h>
#include <pentago/mpi/utility.h>
#include <pentago/filter.h>
#include <pentago/thread.h>
#include <pentago/utility/ceil_div.h>
#include <pentago/utility/char_view.h>
#include <pentago/utility/memory.h>
#include <other/core/array/RawArray.h>
#include <other/core/python/module.h>
#include <snappy.h>
namespace pentago {
namespace mpi {

const int raw_max_fast_compressed_size = 1+snappy::MaxCompressedLength(sizeof(Vector<super_t,2>)*sqr(sqr(block_size)));

int fast_compress(RawArray<Vector<super_t,2>> uncompressed, RawArray<uint8_t> compressed, const event_t event) {
  thread_time_t time(snappy_kind,event);
  // Filter
  if (snappy_filter)
    interleave(uncompressed);
  // Compress
  const size_t input_size = sizeof(Vector<super_t,2>)*uncompressed.size();
  OTHER_ASSERT((size_t)compressed.size()>=1+snappy::MaxCompressedLength(input_size));
  size_t output_size;
  compressed[0] = 1; // Set compressed flag
  snappy::RawCompress((const char*)uncompressed.data(),input_size,(char*)compressed.data()+1,&output_size);
  if (output_size>=input_size) {
    compressed[0] = 0; // Fall back to uncompressed mode
    output_size = input_size;
    memcpy(compressed.data()+1,uncompressed.data(),input_size);
  }
  return output_size+1;
}

// Same as fast_uncompress, but doesn't require an exact size match in the uncompressed buffer and returns the correct size.
static inline int fast_uncompress_helper(RawArray<const uint8_t> compressed, RawArray<Vector<super_t,2>> uncompressed, const event_t event) {
  const auto n = sizeof(Vector<super_t,2>);
  thread_time_t time(unsnappy_kind,event);
  // Uncompress
  OTHER_ASSERT(compressed.size());
  int count;
  if (compressed[0]) { // Compressed mode
    size_t uncompressed_size;
    OTHER_ASSERT(snappy::GetUncompressedLength((const char*)compressed.data()+1,compressed.size()-1,&uncompressed_size));
    if (uncompressed_size > memory_usage(uncompressed))
      die("fast_uncompress: expected size at most %zu, got %zu, event 0x%llx",(size_t)uncompressed.size(),uncompressed_size,event);
    if (uncompressed_size & (n-1))
      die("fast_uncompress: expected size a multiple of %zu, got %zu, event 0x%llx",n,uncompressed_size,event);
    OTHER_ASSERT(snappy::RawUncompress((const char*)compressed.data()+1,compressed.size()-1,(char*)uncompressed.data()));
    count = uncompressed_size/n;
  } else {
    OTHER_ASSERT((size_t)compressed.size()<=1+memory_usage(uncompressed) && !((compressed.size()-1)&(n-1)));
    memcpy(uncompressed.data(),compressed.data()+1,memory_usage(uncompressed));
    count = compressed.size()/n;
  }
  // Unfilter
  if (snappy_filter)
    uninterleave(uncompressed.slice(0,count));
  return count;
}

void fast_uncompress(RawArray<const uint8_t> compressed, RawArray<Vector<super_t,2>> uncompressed, const event_t event) {
  const int count = fast_uncompress_helper(compressed,uncompressed,event);
  if (count != uncompressed.size())
    die("fast_uncompress: expected count %d, got %d, event 0x%llx",uncompressed.size(),count,event);
}

// Thread local temporary buffer for local compression and decompression.
static inline RawArray<Vector<super_t,2>> local_buffer() {
  const int count = ceil_div(raw_max_fast_compressed_size,sizeof(Vector<super_t,2>));
  static __thread Vector<super_t,2>* buffer = 0;
  if (!buffer)
    buffer = (Vector<super_t,2>*)malloc(sizeof(Vector<super_t,2>)*count);
  return RawArray<Vector<super_t,2>>(count,buffer);
}

RawArray<uint8_t> local_fast_compress(RawArray<Vector<super_t,2>> uncompressed, const event_t event) {
  const auto compressed = char_view(local_buffer());
  return compressed.slice(0,fast_compress(uncompressed,compressed,event));
}
   
RawArray<Vector<super_t,2>> local_fast_uncompress(RawArray<const uint8_t> compressed, const event_t event) {
  const auto uncompressed = local_buffer();
  return uncompressed.slice(0,fast_uncompress_helper(compressed,uncompressed,event));
}

}
}
using namespace pentago::mpi;

void wrap_fast_compress() {
  OTHER_FUNCTION(fast_compress)
  OTHER_FUNCTION(fast_uncompress)
}
