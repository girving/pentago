// Add commas to large integers
#pragma once

#include <cstdint>
#include <string>
namespace pentago {

using std::string;

string large(int64_t x);
string large(uint64_t x);

}
