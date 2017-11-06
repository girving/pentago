// Hashed transposition table

#include "pentago/search/table.h"
#include "pentago/search/stat.h"
#include "pentago/base/hash.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/array.h"
#include "pentago/utility/popcount.h"
#include "pentago/utility/log.h"
#include "pentago/utility/str.h"
namespace pentago {

using std::ostream;
using std::make_tuple;

// The current transposition table
static int table_bits = 0;
static Array<uint64_t> table;
static table_type_t table_type;

void init_table(int bits) {
  if (bits<1 || bits>30)
    THROW(ValueError,"expected 1<=bits<=30, got bits = %d",bits);
  if (64-bits+score_bits>64)
    THROW(ValueError,"bits = %d is too small, the high order hash bits won't fit",bits);
  table_bits = bits;
  slog("initializing table: bits = %d, size = %gMB", bits, pow(2.,double(bits-20+3)));
  table = Array<uint64_t>(); // Allocate memory lazily in set_table_type
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
    THROW(RuntimeError,"transposition table not initialized: table_bits = %d",table_bits);
  if (table_type==blank_table)
    table_type = type;
  if (table_type!=type)
    THROW(RuntimeError,"transposition table already set to type %s, must reinitialize before changing to type %s",str(table_type),str(type));

  // Allocate table if we haven't already
  GEODE_ASSERT(!table.size() || table.size()==(1<<table_bits));
  if (!table.size())
    table = Array<uint64_t>(1<<table_bits);
}

score_t lookup(board_t board) {
  assert(table_type!=blank_table);
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
  assert(table_type!=blank_table);
  uint64_t h = hash_board(board);
  uint64_t& entry = table[h&((1<<table_bits)-1)];
  if (entry>>score_bits==h>>table_bits || uint16_t(entry&score_mask)>>2 <= score>>2)
    entry = h>>table_bits<<score_bits|score;
}

tuple<Array<board_t>,Array<score_t>> read_table(int max_count, int min_depth) {
  GEODE_ASSERT(table_bits>=10 && table_type!=blank_table);
  const int size = (1<<table_bits)-1;
  GEODE_ASSERT(table.size()==size);
  Array<board_t> boards(size, uninit);
  Array<score_t> scores(size, uninit);
  for (const int h : range(size)) {
    const uint64_t entry = table[h];
    if (!entry)
      continue;
    const score_t score = entry&score_mask;
    if (score>>2 < min_depth)
      continue;
    const board_t board = inverse_hash_board(entry>>score_bits<<table_bits|h);
    if (count_stones(board) > max_count)
      continue;
    boards[h] = board;
    scores[h] = score;
  }
  return make_tuple(boards,scores);
}

}
