// 64-bit board hash
#pragma once

#include "pentago/base/board.h"
namespace pentago {

static inline uint64_t hash_board(board_t key) {
  // Invertible hash function from http://www.concentric.net/~ttwang/tech/inthash.htm
  key = (~key) + (key << 21); // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8); // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4); // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}

// The inverse of hash_board (for testing and error reporting purposes)
uint64_t inverse_hash_board(uint64_t key);

}
