// uint128_t
#pragma once

#include <type_traits>
#include <cstdint>
#include <string>
namespace pentago {

using std::string;

// We assume a native __uint128_t type
typedef __uint128_t uint128_t;

string str(uint128_t n);

}

namespace std {
ostream& operator<<(ostream& output, pentago::uint128_t n);
}
