// Unified interface to compression libraries
#pragma once

#include <geode/array/Array.h>
#include <pentago/utility/thread.h>
#include <boost/function.hpp>
namespace pentago {

using namespace geode;
using boost::function;

GEODE_EXPORT Array<uint8_t> compress(RawArray<const uint8_t> data, int level, event_t event);
GEODE_EXPORT Array<uint8_t> decompress(RawArray<const uint8_t> compressed, size_t uncompressed_size, event_t event);

GEODE_EXPORT size_t compress_memusage(int level);

}
