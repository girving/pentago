// High level interface to board

#include "pentago/high/board.h"
#include "pentago/base/score.h"
#include "pentago/base/symmetry.h"
#include "pentago/mid/halfsuper.h"
#ifndef __wasm__
#include "pentago/search/superengine.h"
#include "pentago/utility/random.h"
#endif  // !__wasm__
NAMESPACE_PENTAGO

using std::max;

// We don't use this very often, so implement it on top of an expensive, hot routine
__attribute__((cold)) static bool slow_won(const side_t side) {
  return halfsuper_wins(side, 0)[0];
}

bool high_board_t::done() const {
  return slow_won(side(0)) || slow_won(side(1)) || ply_ == 2*36;
}

high_board_t high_board_t::place(const int bit) const {
  const auto move = side_t(1) << bit;
  NON_WASM_ASSERT(!middle() && (move & empty_mask()));
  side_t after[2] = {side(0), side(1)};
  after[turn()] |= move;
  return high_board_t(after[0], after[1], ply_ + 1);
}

high_board_t high_board_t::place(const int x, const int y) const {
  NON_WASM_ASSERT(unsigned() < 6 && unsigned(y) < 6);
  return place(3*(x%3) + y%3 + 16*(2*(x/3) + y/3));
}

// Avoid a dependence on general board transformation for wasm
// TODO: Can we compress this using bit twiddling trickery?
static quadrant_t slow_rotate_quadrant_side(const quadrant_t side, const int d) {
  quadrant_t result = 0;
  for (const int x : range(3))
    for (const int y : range(3))
      if (side & 1<<(3*x+y))
        result |= 1<<(d==1 ? 3*(2-y)+x : 3*y+2-x);
  return result;
}

high_board_t high_board_t::rotate(const int q, const int d) const {
  NON_WASM_ASSERT(middle() && 0 <= q && q < 4 && d==1 || d==-1);
  side_t after[2] = {side(0), side(1)};
  for (const int s : range(2)) {
    const auto old = quadrant(after[s], q);
    after[s] ^= side_t(old ^ slow_rotate_quadrant_side(old, d)) << 16*q;
  }
  return high_board_t(after[0], after[1], ply_ + 1);
}

int high_board_t::immediate_value() const {
  NON_WASM_ASSERT(done());
  const bool bw = slow_won(side(0)),
             ww = slow_won(side(1));
  if (bw || ww)
    return bw && ww ? 0 : bw==!turn() ? 1 : -1;
  return 0;
}

#ifndef __wasm__
high_board_t high_board_t::from_board(const board_t board, const bool middle) {
  const auto side0 = unpack(board, 0);
  const auto side1 = unpack(board, 1);
  return high_board_t(side0, side1, 2*popcount(side0 | side1) - middle);
}

Array<const high_board_t> high_board_t::moves() const {
  vector<high_board_t> moves;
  if (!middle()) { // Place a stone
    const auto empty = empty_mask();
    for (const int bit : range(64))
      if (empty & side_t(1)<<bit)
        moves.push_back(place(bit));
  } else { // Rotate a quadrant
    for (const int q : range(4))
      for (const int d : {-1, 1})
        moves.push_back(rotate(q, d));
  }
  return asarray(moves).copy();;
}

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
    const auto flip = flip_board(board(), turn());
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
    const auto board = high_board_t::from_board(transform_board(symmetry_t(local_symmetry_t(r)), boards[i]), false);
    const int value = board.value_check(cache);
    GEODE_ASSERT(value == wins[i][n&1](r)-wins[i][!(n&1)](r));
    counts[value+1]++;
  }
  return counts;
}

string high_board_t::name() const {
  return format("%d%s", board(), middle() ? "m" : "");
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
      return high_board_t::from_board(board, left==1);
  }
  THROW(ValueError, "high_board_t: invalid board name '%s'", name);
}
#endif  // !__wasm__

END_NAMESPACE_PENTAGO
