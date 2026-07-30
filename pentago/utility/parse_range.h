#pragma once

// Separate from range.h so that range.h stays usable in the freestanding
// wasm build (web/client/build-wasm), which lacks <string>.

#include "pentago/utility/range.h"
#include <string>
namespace pentago {

using std::string;

// Parse a Python-style range string "lo:hi", ":hi", "lo:", or ":" into Range<int>.
// Missing lo defaults to 0, missing hi defaults to total.
Range<int> parse_range(const string& s, const int total);

}
