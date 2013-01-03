// Interface to snappy

#include <pentago/mpi/fast_compress.h>
#include <pentago/mpi/config.h>
#include <pentago/mpi/utility.h>
#include <pentago/filter.h>
#include <pentago/thread.h>
#include <pentago/utility/memory.h>
#include <other/core/array/RawArray.h>
#include <other/core/python/module.h>
#include <snappy.h>
namespace pentago {
namespace mpi {

const int raw_max_fast_compressed_size = 1+snappy::MaxCompressedLength(sizeof(Vector<super_t,2>)*sqr(sqr(block_size)));

int fast_compress(RawArray<Vector<super_t,2>> uncompressed, RawArray<uint8_t> compressed, event_t event) {
  thread_time_t time(snappy_kind,event);
  // Filter
#if PENTAGO_MPI_SNAPPY_FILTER
  interleave(uncompressed);
#endif
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

void fast_uncompress(RawArray<const uint8_t> compressed, RawArray<Vector<super_t,2>> uncompressed, event_t event) {
  thread_time_t time(unsnappy_kind,event);
  // Uncompress
  OTHER_ASSERT(compressed.size());
  if (compressed[0]) { // Compressed mode
    size_t uncompressed_size;
    OTHER_ASSERT(snappy::GetUncompressedLength((const char*)compressed.data()+1,compressed.size()-1,&uncompressed_size));
    if (uncompressed_size != memory_usage(uncompressed))
      die("fast_uncompress: expected size %zu, got %zu, event 0x%llx",uncompressed.size(),uncompressed_size,event);
    OTHER_ASSERT(snappy::RawUncompress((const char*)compressed.data()+1,compressed.size()-1,(char*)uncompressed.data()));
  } else {
    OTHER_ASSERT((size_t)compressed.size()==1+memory_usage(uncompressed));
    memcpy(uncompressed.data(),compressed.data()+1,memory_usage(uncompressed));
  }
  // Unfilter
#if PENTAGO_MPI_SNAPPY_FILTER
  uninterleave(uncompressed);
#endif
}

}
}
using namespace pentago::mpi;

void wrap_fast_compress() {
  OTHER_FUNCTION(fast_compress)
  OTHER_FUNCTION(fast_uncompress)
}
