#include "pentago/utility/portable_hash.h"
namespace pentago {

// We use a modified version of https://github.com/CTrabant/teeny-sha1,
// modified from https://github.com/CTrabant/teeny-sha1/commit/f05a2d0019e12850bb482274c1a21ae7a4fb84bd.

/*******************************************************************************
 * Teeny SHA-1
 *
 * The below sha1digest() calculates a SHA-1 hash value for a
 * specified data buffer and generates a hex representation of the
 * result.  This implementation is a re-forming of the SHA-1 code at
 * https://github.com/jinqiangshou/EncryptionLibrary.
 *
 * Copyright (c) 2017 CTrabant
 *
 * License: MIT, see included LICENSE file for details.
 *
 * To use the sha1digest() function either copy it into an existing
 * project source code file or include this file in a project and put
 * the declaration (example below) in the sources files where needed.
 ******************************************************************************/

#define SHA1ROTATELEFT(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

string sha1(RawArray<const uint8_t> data) {
  const uint32_t databytes = data.size();
  const uint64_t databits = ((uint64_t)databytes) * 8;
  const uint32_t loopcount = (databytes + 8) / 64 + 1;
  const uint32_t tailbytes = 64 * loopcount - databytes;

  /* Pre-processing of data tail (includes padding to fill out 512-bit chunk):
     Add bit '1' to end of message (big-endian)
     Add 64-bit message length in bits at very end (big-endian) */
  uint8_t datatail[128] = {0};
  datatail[0] = 0x80;
  datatail[tailbytes - 8] = uint8_t(databits >> 56 & 0xFF);
  datatail[tailbytes - 7] = uint8_t(databits >> 48 & 0xFF);
  datatail[tailbytes - 6] = uint8_t(databits >> 40 & 0xFF);
  datatail[tailbytes - 5] = uint8_t(databits >> 32 & 0xFF);
  datatail[tailbytes - 4] = uint8_t(databits >> 24 & 0xFF);
  datatail[tailbytes - 3] = uint8_t(databits >> 16 & 0xFF);
  datatail[tailbytes - 2] = uint8_t(databits >> 8 & 0xFF);
  datatail[tailbytes - 1] = uint8_t(databits >> 0 & 0xFF);

  /* Process each 512-bit chunk */
  uint32_t didx = 0;
  uint32_t H[5] = {0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0};
  for (uint32_t lidx = 0; lidx < loopcount; lidx++) {
    /* Compute all elements in W */
    uint32_t W[80] = {0};

    /* Break 512-bit chunk into sixteen 32-bit, big endian words */
    for (uint32_t widx = 0; widx <= 15; widx++) {
      int32_t wcount = 24;

      /* Copy byte-per byte from specified buffer */
      while (didx < databytes && wcount >= 0) {
        W[widx] += (((uint32_t)data[didx]) << wcount);
        didx++;
        wcount -= 8;
      }
      /* Fill out W with padding as needed */
      while (wcount >= 0) {
        W[widx] += (((uint32_t)datatail[didx - databytes]) << wcount);
        didx++;
        wcount -= 8;
      }
    }

    /* Extend the sixteen 32-bit words into eighty 32-bit words, with potential optimization from:
       "Improving the Performance of the Secure Hash Algorithm (SHA-1)" by Max Locktyukhin */
    for (uint32_t widx = 16; widx <= 31; widx++)
      W[widx] = SHA1ROTATELEFT((W[widx - 3] ^ W[widx - 8] ^ W[widx - 14] ^ W[widx - 16]), 1);
    for (uint32_t widx = 32; widx <= 79; widx++)
      W[widx] = SHA1ROTATELEFT((W[widx - 6] ^ W[widx - 16] ^ W[widx - 28] ^ W[widx - 32]), 2);

    /* Main loop */
    uint32_t a = H[0];
    uint32_t b = H[1];
    uint32_t c = H[2];
    uint32_t d = H[3];
    uint32_t e = H[4];

    for (uint32_t idx = 0; idx <= 79; idx++) {
      uint32_t f = 0, k = 0;
      if (idx <= 19) {
        f = (b & c) | ((~b) & d);
        k = 0x5A827999;
      } else if (idx >= 20 && idx <= 39) {
        f = b ^ c ^ d;
        k = 0x6ED9EBA1;
      } else if (idx >= 40 && idx <= 59) {
        f = (b & c) | (b & d) | (c & d);
        k = 0x8F1BBCDC;
      } else if (idx >= 60 && idx <= 79) {
        f = b ^ c ^ d;
        k = 0xCA62C1D6;
      }
      const uint32_t temp = SHA1ROTATELEFT(a, 5) + f + e + k + W[idx];
      e = d;
      d = c;
      c = SHA1ROTATELEFT(b, 30);
      b = a;
      a = temp;
    }

    H[0] += a;
    H[1] += b;
    H[2] += c;
    H[3] += d;
    H[4] += e;
  }

  /* Format digest */
  return format("%08x%08x%08x%08x%08x", H[0], H[1], H[2], H[3], H[4]);
}

}  // namespace pentago
