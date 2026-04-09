// Streaming GCS reader with multi-threaded readahead
//
// gcs_streamer_t: streams requested byte ranges from a GCS object sequentially.
// supertensor_stream_t: wraps gcs_streamer_t with supertensor format knowledge,
// yielding compressed blocks in file-offset order for parallel decompression.
#pragma once

#include "pentago/base/section.h"
#include "pentago/base/superscore.h"
#include "pentago/utility/array.h"
#include "pentago/utility/noncopyable.h"
#include "pentago/utility/vector.h"
#include <functional>
#include <memory>
#include <optional>
namespace pentago {

using std::function;
using std::optional;
using std::unique_ptr;

// Fetch a byte range from a file. Returns up to `size` bytes starting at `offset`
// (fewer at EOF). Used by streamer_t for both local and GCS backends.
using fetch_fn_t = function<Array<const uint8_t>(int64_t offset, int64_t size)>;

// Stream requested byte ranges from a file, in file-offset order.
// Multiple reader threads fetch sequential chunks with readahead.
// Works for both local files and GCS objects via the fetch callback.
class streamer_t : noncopyable_t {
public:
  struct request_t { int64_t offset; int32_t size; int id; };

  struct result_t {
    int id = -1;
    Array<const uint8_t> data;
    explicit operator bool() const { return id >= 0; }
  };

  // Requests are sorted internally by offset. Each request must fit
  // within a single chunk (i.e. request.size <= chunk_bytes).
  streamer_t(const fetch_fn_t& fetch, RawArray<const request_t> requests,
             const int64_t chunk_bytes, const int64_t readahead_bytes, const int num_threads);
  ~streamer_t();

  // Get next requested range. Thread-safe, multi-consumer.
  // Returns result with id=-1 when all requests consumed.
  result_t next();

private:
  struct impl_t;
  unique_ptr<impl_t> impl;
};

// Stream compressed blocks from a supertensor file (local or GCS).
// Phase 1 (constructor): reads header + index.
// Phase 2 (next): streams blocks in file-offset order via streamer_t.
class supertensor_stream_t : noncopyable_t {
public:
  struct block_t {
    section_t section;
    Vector<uint8_t,4> block;
    int filter;
    Array<const uint8_t> compressed;
    explicit operator bool() const { return compressed.size() > 0; }
  };

  supertensor_stream_t(const string& path, int64_t readahead_bytes, int num_threads);
  ~supertensor_stream_t();

  int64_t total_blocks() const;

  // Get next compressed block in file-offset order.
  // Thread-safe, multi-consumer. Returns empty block when done.
  block_t next();

  // Decompress a block returned by next() (call from any thread).
  static Array<Vector<super_t,2>,4> decompress(const block_t& block);

private:
  struct impl_t;
  unique_ptr<impl_t> impl;
};

}  // namespace pentago
