// Import tinyformat::format into the pentago namespace
#pragma once

#ifndef __EMSCRIPTEN__
#include "tinyformat.h"
namespace pentago {

using tinyformat::format;

}
#endif
