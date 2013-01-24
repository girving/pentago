// Add commas to large integers
#pragma once

#include <other/core/utility/config.h>
#include <stdint.h>
#include <string>
namespace pentago {

using std::string;

OTHER_EXPORT string large(int64_t x);
OTHER_EXPORT string large(uint64_t x);

}
