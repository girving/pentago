// Unified interface to compression libraries

#include <other/core/array/Array.h>
#include <pentago/thread.h>
#include <boost/function.hpp>
namespace pentago {

using namespace other;
using boost::function;

OTHER_EXPORT Array<uint8_t> compress(RawArray<const uint8_t> data, int level, event_t event);
OTHER_EXPORT Array<uint8_t> decompress(Array<const uint8_t> compressed, size_t uncompressed_size, event_t event);

OTHER_EXPORT size_t compress_memusage(int level);

}
