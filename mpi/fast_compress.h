// Interface to snappy
#pragma once

#include <pentago/superscore.h>
#include <pentago/thread.h>
namespace pentago {
namespace mpi {

// We modify the snappy format to expand by at most 1 byte.  This number is the maximum size before we
// fix the format, since snappy's compression routine doesn't take an output limit.
extern const int raw_max_fast_compressed_size;

// Compress an array and return the size of the result.  The input is destroyed.
int fast_compress(RawArray<Vector<super_t,2>> uncompressed, RawArray<uint8_t> compressed, event_t event);

// Uncompress an array
void fast_uncompress(RawArray<const uint8_t> compressed, RawArray<Vector<super_t,2>> uncompressed, event_t event);

}
}
