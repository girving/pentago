// Interface to snappy
#pragma once

#include <pentago/superscore.h>
#include <pentago/thread.h>
namespace pentago {
namespace mpi {

// Bound the maximum compressed length.
extern const int max_fast_compressed_size;

// Compress an array and return the size of the result.  The input is destroyed.
int fast_compress(RawArray<Vector<super_t,2>> uncompressed, RawArray<char> compressed, event_t event);

// Uncompress an array
void fast_uncompress(RawArray<const char> compressed, RawArray<Vector<super_t,2>> uncompressed, event_t event);

}
}
