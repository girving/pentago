// Hashed transposition table

#include "table.h"
#include "stat.h"
#include <other/core/array/Array.h>
#include <other/core/math/popcount.h>
#include <other/core/python/module.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/format.h>
#include <other/core/utility/str.h>
namespace pentago {

using namespace other;
using std::ostream;
using std::cout;
using std::endl;

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

// The inverse of hash_board (for testing purposes)
static inline uint64_t inverse_hash_board(uint64_t key) {
  uint64_t tmp;

  // Invert key = key + (key << 31)
  tmp = key-(key<<31);
  key = key-(tmp<<31);

  // Invert key = key ^ (key >> 28)
  tmp = key^key>>28;
  key = key^tmp>>28;

  // Invert key *= 21
  key *= 14933078535860113213u;

  // Invert key = key ^ (key >> 14)
  tmp = key^key>>14;
  tmp = key^tmp>>14;
  tmp = key^tmp>>14;
  key = key^tmp>>14;

  // Invert key *= 265
  key *= 15244667743933553977u;

  // Invert key = key ^ (key >> 24)
  tmp = key^key>>24;
  key = key^tmp>>24;

  // Invert key = (~key) + (key << 21)
  tmp = ~key;
  tmp = ~(key-(tmp<<21));
  tmp = ~(key-(tmp<<21));
  key = ~(key-(tmp<<21));

  return key;
}

// The current transposition table
static int table_bits = 0;
static Array<uint64_t> table;
static table_type_t table_type;

void init_table(int bits) {
  if (bits<1 || bits>30)
    throw ValueError(format("expected 1<=bits<=30, got bits = %d",bits));
  if (64-bits+score_bits>64)
    throw ValueError(format("bits = %d is too small, the high order hash bits won't fit",bits));
  table_bits = bits;
  cout << "initializing table: bits = "<<bits<<", size = "<<pow(2.,double(bits-20+3))<<"MB"<<endl;
  Array<uint64_t>((uint64_t)1<<bits).swap(table);
  table_type = blank_table;
}

static ostream& operator<<(ostream& output, table_type_t type) {
  return output<<(type==blank_table?"blank"
                 :type==normal_table?"normal"
                 :type==simple_table?"simple"
                 :"unknown");
}

void set_table_type(table_type_t type) {
  if (table_bits<10)
    throw RuntimeError(format("transposition table not initialized: table_bits = %d",table_bits));
  if (table_type==blank_table)
    table_type = type;
  if (table_type!=type)
    throw RuntimeError(format("transposition table already set to type %s, must reinitialize before changing to type %s",str(table_type),str(type)));
}

score_t lookup(board_t board) {
  STAT(total_lookups++);
  uint64_t h = hash_board(board);
  uint64_t entry = table[h&((1<<table_bits)-1)];
  if (entry>>score_bits==h>>table_bits) {
    STAT(successful_lookups++);
    return entry&score_mask;
  }
  return score(0,1);
}

void store(board_t board, score_t score) {
  uint64_t h = hash_board(board);
  uint64_t& entry = table[h&((1<<table_bits)-1)];
  if (entry>>score_bits==h>>table_bits || uint16_t(entry&score_mask)>>2 <= score>>2)
    entry = h>>table_bits<<score_bits|score;
}

Tuple<Array<board_t>,Array<score_t> > read_table(int max_count, int min_depth) {
  OTHER_ASSERT(table_bits>=10);
  const uint64_t size = (1<<table_bits)-1;
  Array<board_t> boards;
  Array<score_t> scores;
  for (uint64_t h=0;h<size;h++) {
    uint64_t entry = table[h];
    if (!entry)
      continue;
    score_t score = entry&score_mask;
    if (score>>2 < min_depth)
      continue;
    board_t board = inverse_hash_board(entry>>score_bits<<table_bits|h);
    if (popcount(unpack(board,0))+popcount(unpack(board,1)) > max_count)
      continue;
    boards.append(board);
    scores.append(score);
  }
  return tuple(boards,scores);
}

}
using namespace pentago;
using namespace other::python;

void wrap_table() {
  OTHER_FUNCTION(hash_board)
  OTHER_FUNCTION(inverse_hash_board)
  OTHER_FUNCTION(init_table)
  OTHER_FUNCTION(read_table)
}
