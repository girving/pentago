// Interface to snappy
#pragma once

#include <pentago/base/superscore.h>
#include <pentago/utility/thread.h>
namespace pentago {
namespace end {

// We modify the snappy format to expand by at most 1 byte.  This number is the maximum size before we
// fix the format, since snappy's compression routine doesn't take an output limit.
extern const int raw_max_fast_compressed_size;

// Compress an array and return the size of the result.  The input is destroyed.
int fast_compress(RawArray<Vector<super_t,2>> uncompressed, RawArray<uint8_t> compressed, const event_t event);

// Uncompress an array
void fast_uncompress(RawArray<const uint8_t> compressed, RawArray<Vector<super_t,2>> uncompressed, const event_t event);

// The following two functions share the same thread local buffer.

// Compress into a thread local buffer.  The input is destroyed.
// The returned view is valid until the next call to either local_fast_compress or local_fast_uncompress.
RawArray<uint8_t> local_fast_compress(RawArray<Vector<super_t,2>> uncompressed, const event_t event);

// Uncompress into a thread local buffer (the same buffer used by fast_uncompress).
// The returned view is valid until the next call to either local_fast_compress or local_fast_uncompress.
OTHER_EXPORT RawArray<Vector<super_t,2>> local_fast_uncompress(RawArray<const uint8_t> compressed, const event_t event);

}
}
