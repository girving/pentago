// Multidimensional superscore filtering to precondition zlib compression
#pragma once

#include "superscore.h"
namespace pentago {

void interleave(RawArray<Vector<super_t,2>> data);
void uninterleave(RawArray<Vector<super_t,2>> data);

}
