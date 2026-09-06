// liblzma configuration for the freestanding wasm decoder build
//
// Stands in for the configure-generated config.h: decoders only, the LZMA2 filter only, CRC32
// and CRC64 checks only, and no threads.  Anything not defined here is compiled out.
#pragma once

#define HAVE_DECODERS 1
#define HAVE_DECODER_LZMA2 1
#define HAVE_CHECK_CRC32 1
#define HAVE_CHECK_CRC64 1
#define HAVE_STDBOOL_H 1
#define HAVE_STDINT_H 1
#define HAVE___BUILTIN_BSWAPXX 1
#define TUKLIB_FAST_UNALIGNED_ACCESS 1
#define TUKLIB_SYMBOL_PREFIX lzma_
#define SIZEOF_SIZE_T 4
#define NDEBUG 1
