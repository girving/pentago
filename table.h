// Hashed transposition table
#pragma once

// Notes:
// 1. Collisions are resolved simply: the entry with greater depth wins.
// 2. Since our hash is bijective, we store only the high order bits of the hash to detect collisions.
//    This is compatible with zero initialization since hash_board(board)!=0 for all valid boards.

#include <pentago/board.h>
#include <pentago/score.h>
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
extern uint64_t inverse_hash_board(uint64_t key);

// Initialize a empty table with size 1<<bits entries
extern void init_table(int bits);

// Different kinds of tables
enum table_type_t {blank_table,normal_table,simple_table};

// Freeze the table into the given type
extern void set_table_type(table_type_t type);

// Lookup an entry in the table, returning a depth 0 tie (score(0,1)) if nothing is found
extern score_t lookup(board_t board);

// Store a new entry
extern void store(board_t board, score_t score);

}
