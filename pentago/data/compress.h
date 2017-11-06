// Unified interface to compression libraries
#pragma once

#include "pentago/utility/array.h"
#include "pentago/utility/thread.h"
namespace pentago {

Array<uint8_t> compress(RawArray<const uint8_t> data, int level, event_t event);
Array<uint8_t> decompress(RawArray<const uint8_t> compressed, size_t uncompressed_size, event_t event);

size_t compress_memusage(int level);

}
