// Hashed transposition table

#include "table.h"
#include "stat.h"
#include <other/core/array/Array.h>
#include <other/core/python/module.h>
#include <other/core/utility/format.h>
#include <other/core/utility/str.h>
namespace pentago {

using namespace other;
using std::ostream;

static inline uint64_t hash(board_t key) {
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
  uint64_t h = hash(board);
  uint64_t entry = table[h&((1<<table_bits)-1)];
  if (entry>>score_bits==h>>table_bits) {
    STAT(successful_lookups++);
    return entry&score_mask;
  }
  return score(0,1);
}

void store(board_t board, score_t score) {
  uint64_t h = hash(board);
  uint64_t& entry = table[h&((1<<table_bits)-1)];
  if (entry>>score_bits==h>>table_bits || uint16_t(entry&score_mask)>>2 <= score>>2)
    entry = h>>table_bits<<score_bits|score;
}

}
using namespace pentago;
using namespace other::python;

void wrap_table() {
  OTHER_FUNCTION(init_table)
}
