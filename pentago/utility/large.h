// Add commas to large integers
#pragma once

#include <geode/utility/config.h>
#include <stdint.h>
#include <string>
namespace pentago {

using std::string;

GEODE_EXPORT string large(int64_t x);
GEODE_EXPORT string large(uint64_t x);

}
