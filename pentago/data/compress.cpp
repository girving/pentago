// Unified interface to compression libraries

#include <pentago/data/compress.h>
#include <pentago/utility/thread.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/debug.h>
#include <zlib.h>
#include <lzma.h>
namespace pentago {

static const char* zlib_error(int z) {
  return z==Z_MEM_ERROR?"out of memory"
       : z==Z_BUF_ERROR?"insufficient output buffer space"
       : z==Z_DATA_ERROR?"incomplete or corrupted data"
       : z==Z_STREAM_ERROR?"invalid level"
       : "unknown error";
}

static const char* lzma_error(lzma_ret r) {
  switch (r) {
    case LZMA_STREAM_END:        return "end of stream was reached";
    case LZMA_NO_CHECK:          return "input stream has no integrity check";
    case LZMA_UNSUPPORTED_CHECK: return "cannot calculate the integrity check";
    case LZMA_GET_CHECK:         return "integrity check is now available";
    case LZMA_MEM_ERROR:         return "cannot allocate memory";
    case LZMA_MEMLIMIT_ERROR:    return "memory usage limit was reached";
    case LZMA_FORMAT_ERROR:      return "file format not recognized";
    case LZMA_OPTIONS_ERROR:     return "invalid or unsupported options";
    case LZMA_DATA_ERROR:        return "data is corrupt";
    case LZMA_BUF_ERROR:         return "no progress is possible";
    case LZMA_PROG_ERROR:        return "programming error";
    default:                     return "unknown error";
  }
}

static bool is_lzma(RawArray<const uint8_t> data) {
  static const uint8_t magic[6] = {0xfd,'7','z','X','Z',0};
  return data.size()>=6 && !memcmp(data.data(),magic,6);
}

Array<uint8_t> compress(RawArray<const uint8_t> data, int level, event_t event) {
  thread_time_t time(compress_kind,event);
  if (level<20) { // zlib
    size_t dest_size = compressBound(data.size());
    Array<uint8_t> compressed(CHECK_CAST_INT(dest_size),uninit);
    int z = compress2(compressed.data(),&dest_size,(uint8_t*)data.data(),data.size(),level);
    if (z!=Z_OK)
      THROW(IOError,"zlib failure in compress_and_write: %s",zlib_error(z));
    return compressed.slice_own(0,CHECK_CAST_INT(dest_size));
  } else { // lzma
    size_t dest_size = lzma_stream_buffer_bound(data.size());
    Array<uint8_t> compressed(CHECK_CAST_INT(dest_size),uninit);
    size_t pos = 0;
    lzma_ret r = lzma_easy_buffer_encode(level-20,LZMA_CHECK_CRC64,0,data.data(),data.size(),compressed.data(),&pos,dest_size);
    if (r!=LZMA_OK)
      THROW(RuntimeError,"lzma compression error: %s (%d)",lzma_error(r),r);
    return compressed.slice_own(0,CHECK_CAST_INT(pos));
  }
}

size_t compress_memusage(int level) {
  if (level<20) { // zlib
    GEODE_ASSERT(1<=level && level<=MAX_MEM_LEVEL);
    return (1<<(MAX_WBITS+2))+(1<<(level+9));
  } else // lzma
    return lzma_easy_encoder_memusage(level-20);
}

Array<uint8_t> decompress(RawArray<const uint8_t> compressed, const size_t uncompressed_size, event_t event) {
  GEODE_ASSERT(uncompressed_size<(uint64_t)1<<31);
  thread_time_t time(decompress_kind,event);
  size_t dest_size = uncompressed_size;
  Array<uint8_t> uncompressed = aligned_buffer<uint8_t>(CHECK_CAST_INT(dest_size));
  if (!is_lzma(compressed)) { // zlib
    int z = uncompress((uint8_t*)uncompressed.data(),&dest_size,compressed.data(),compressed.size());
    if (z!=Z_OK)
      THROW(IOError,"zlib failure in read_and_uncompress: %s",zlib_error(z));
  } else { // lzma
    const uint32_t flags = LZMA_TELL_NO_CHECK | LZMA_TELL_UNSUPPORTED_CHECK;
    uint64_t memlimit = UINT64_MAX;
    size_t in_pos = 0, out_pos = 0;
    lzma_ret r = lzma_stream_buffer_decode(&memlimit,flags,0,compressed.data(),&in_pos,compressed.size(),uncompressed.data(),&out_pos,dest_size);
    if (r!=LZMA_OK)
      THROW(IOError,"lzma failure in read_and_uncompress: %s (%d)",lzma_error(r),r);
  }
  if (dest_size != uncompressed_size)
    THROW(IOError,"read_and_compress: expected uncompressed size %zu, got %zu",uncompressed_size,dest_size);
  return uncompressed;
}

}
