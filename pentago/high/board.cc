// High level interface to board

#include "pentago/high/board.h"
#include "pentago/base/score.h"
#include "pentago/base/symmetry.h"
#ifndef __wasm__
#include "pentago/search/superengine.h"
#include "pentago/utility/random.h"
#endif  // !__wasm__
namespace pentago {

using std::max;

high_board_t::high_board_t(const board_t board, const bool middle)
  : rep_{uint32_t( middle ? ~board : board),
         uint32_t((middle ? ~board : board) >> 32)} {
  // Check correctness
  check_board(board);
  const auto [side0, side1] = slow_unpack(board);
  const int count0 = popcount(side0),
            count1 = popcount(side1);
  GEODE_ASSERT(count0 + count1 == slow_count_stones(board));
  const auto turn = this->turn();
  if (count0-turn-middle*(turn==0)!=count1-middle*(turn==1) || (middle && !board))
    THROW(ValueError, "high_board_t: inconsistent board %lld, turn %d, middle %d, "
          "side counts %d %d", board, turn, middle, count0, count1);
}

high_board_t::~high_board_t() {}

int high_board_t::count() const {
  return slow_count_stones(board());
}

bool high_board_t::done() const {
  const auto board = this->board();
  const auto [side0, side1] = slow_unpack(board);
  return slow_won(side0)
      || slow_won(side1)
      || (!middle() && slow_count_stones(board)==36);
}

int high_board_t::turn() const {
  const auto board = this->board();
  const auto [side0, side1] = slow_unpack(board);
  const int count0 = popcount(side0),
            count1 = popcount(side1);
  return count0 - 1 == count1 - middle();
}

pile<high_board_t,36> high_board_t::moves() const {
  pile<high_board_t,36> moves;
  if (!middle()) { // Place a stone
    const auto board = this->board();
    const auto [side0, side1] = slow_unpack(board);
    const auto empty = ~(side0 | side1);
    for (const int x : range(6))
      for (const int y : range(6))
        if (empty & board_t(1) << (32*(x/3)+16*(y/3)+3*(x%3)+(y%3)))
          moves.append(place(x, y));
  } else { // Rotate a quadrant
    for (const int qx : range(2))
      for (const int qy : range(2))
        for (const int d : vec(-1,1))
          moves.append(rotate(qx, qy, d));
  }
  return moves;
}

high_board_t high_board_t::place(const int x, const int y) const {
  GEODE_ASSERT(!middle());
  GEODE_ASSERT(0<=x && x<6);
  GEODE_ASSERT(0<=y && y<6);
  const auto board = this->board();
  const auto [side0, side1] = slow_unpack(board);
  const auto empty = ~(side0 | side1);
  const auto move = side_t(1<<(3*(x%3)+y%3))<<16*(2*(x/3)+(y/3));
  GEODE_ASSERT(empty & move);
  return high_board_t(board + slow_flip_board(slow_pack(move, side_t(0)), turn()), true);
}

high_board_t high_board_t::rotate(const int qx, const int qy, const int d) const {
  GEODE_ASSERT(middle());
  GEODE_ASSERT(qx==0 || qx==1);
  GEODE_ASSERT(qy==0 || qy==1);
  GEODE_ASSERT(d==1 || d==-1);
  return high_board_t(slow_transform_board(symmetry_t(local_symmetry_t((d>0?1:3)<<2*(2*qx+qy))), board()), false);
}

int high_board_t::immediate_value() const {
  GEODE_ASSERT(done());
  const auto board = this->board();
  const auto [side0, side1] = slow_unpack(board);
  const bool bw = slow_won(side0),
             ww = slow_won(side1);
  if (bw || ww)
    return bw && ww ? 0 : bw==!turn() ? 1 : -1;
  return 0;
}

#ifndef __wasm__
// If aggressive is true, true is win, otherwise true is win or tie.
static bool exact_lookup(const bool aggressive, const board_t board, const block_cache_t& cache) {
  {
    super_t wins;
    if (cache.lookup(aggressive,board,wins))
      return wins(0);
  }
  // Fall back to tree search
  const score_t score = super_evaluate(aggressive,100,board,Vector<int,4>());
  return (score&3)>=aggressive+1;
}

int high_board_t::value(const block_cache_t& cache) const {
  if (done())
    return immediate_value();
  else if (!middle()) {
    // If we're not at a half move, look up the result
    const auto flip = slow_flip_board(board(), turn());
    return exact_lookup(true,flip,cache)+exact_lookup(false,flip,cache)-1;
  } else {
    int best = -1;
    for (const auto& move : moves()) {
      best = max(best, -move.value(cache));
      if (best==1)
        break;
    }
    return best;
  }
}

int high_board_t::value_check(const block_cache_t& cache) const {
  GEODE_ASSERT(!middle());
  if (done())
    return immediate_value();
  {
    int value = -1;
    for (const auto& move : moves())
      value = max(value, move.value(cache));
    const int lookup = this->value(cache);
    if (value != lookup)
      THROW(AssertionError, "high_board_t.value_check: board %lld, turn %d, middle %d, computed %d != lookup %d",
            board(), turn(), middle(), value, lookup);
    return value;
  }
}

Vector<int,3> high_board_t::sample_check(const block_cache_t& cache, RawArray<const board_t> boards,
                                         RawArray<const Vector<super_t,2>> wins) {
  GEODE_ASSERT(boards.size() == wins.size());
  Vector<int,3> counts;
  Random random(152311341);
  for (const int i : range(boards.size())) {
    const int r = random.bits<uint8_t>();
    const int n = count_stones(boards[i]);
    const high_board_t board(transform_board(symmetry_t(local_symmetry_t(r)), boards[i]), false);
    const int value = board.value_check(cache);
    GEODE_ASSERT(value == wins[i][n&1](r)-wins[i][!(n&1)](r));
    counts[value+1]++;
  }
  return counts;
}

string high_board_t::name() const {
  // Use snprintf to avoid tinyformat dependency under emscripten
  char buffer[24];
  snprintf(buffer, sizeof(buffer), "%llu%s", (unsigned long long)board(), middle() ? "m" : "");
  return buffer;
}

ostream& operator<<(ostream& output, const high_board_t board) {
  output << board.board();
  if (board.middle())
    output << 'm';
  return output;
}

high_board_t high_board_t::parse(const string& name) {
  static_assert(sizeof(long long) == 8);
  if (name.size() && isdigit(name[0]) && (isdigit(name.back()) || name.back()=='m')) {
    char* end;
    const board_t board = strtoll(name.c_str(), &end, 0);
    const auto left = name.c_str()+name.size()-end;
    if (left==0 || (left==1 && name.back()=='m'))
      return high_board_t(board, left==1);
  }
  THROW(ValueError, "high_board_t: invalid board name '%s'", name);
}
#endif  // !__wasm__

}
