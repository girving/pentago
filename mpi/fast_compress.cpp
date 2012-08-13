// Interface to snappy

#include <pentago/mpi/fast_compress.h>
#include <pentago/mpi/config.h>
#include <pentago/filter.h>
#include <pentago/thread.h>
#include <pentago/utility/memory.h>
#include <other/core/array/RawArray.h>
#include <snappy.h>
namespace pentago {
namespace mpi {

const int max_fast_compressed_size = snappy::MaxCompressedLength(sizeof(Vector<super_t,2>)*sqr(sqr(block_size)));

int fast_compress(RawArray<Vector<super_t,2>> uncompressed, RawArray<char> compressed) {
  thread_time_t time("snappy");
  // Filter
#if PENTAGO_MPI_SNAPPY_FILTER
  interleave(uncompressed);
#endif
  // Compress
  OTHER_ASSERT((size_t)compressed.size()>=snappy::MaxCompressedLength(uncompressed.size()));
  size_t size;
  snappy::RawCompress((const char*)uncompressed.data(),sizeof(Vector<super_t,2>)*uncompressed.size(),compressed.data(),&size);
  return size;
}

void fast_uncompress(RawArray<const char> compressed, RawArray<Vector<super_t,2>> uncompressed) {
  thread_time_t time("unsnappy");
  // Uncompress
  size_t uncompressed_size;
  OTHER_ASSERT(   snappy::GetUncompressedLength(compressed.data(),compressed.size(),&uncompressed_size)
               && uncompressed_size==memory_usage(uncompressed));
  OTHER_ASSERT(snappy::RawUncompress(compressed.data(),compressed.size(),(char*)uncompressed.data()));
  // Unfilter
#if PENTAGO_MPI_SNAPPY_FILTER
  uninterleave(uncompressed);
#endif
}

}
}
