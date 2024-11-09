// High level interface to board

#include "board.h"
#include "../base/superscore.h"
#include "../base/symmetry.h"
#include "../mid/halfsuper.h"
#ifndef __wasm__
#include "../utility/random.h"
#endif  // !__wasm__
NAMESPACE_PENTAGO

using std::make_tuple;
using std::max;

// We don't use this very often, so implement it on top of an expensive, hot routine
__attribute__((cold)) static bool slow_won(const side_t side) {
  return halfsuper_wins(side, 0)[0];
}

tuple<bool,int> high_board_t::done_and_value() const {
  const bool bw = slow_won(side(0)),
             ww = slow_won(side(1));
  if (bw || ww)
    return make_tuple(true, bw && ww ? 0 : bw==!turn() ? 1 : -1);
  if (s.ply_ == 2*36)
    return make_tuple(true, 0);
  return make_tuple(false, 0);
}

bool high_board_t::done() const {
  return get<0>(done_and_value());
}

int high_board_t::immediate_value() const {
  const auto [done, value] = done_and_value();
  NON_WASM_ASSERT(done);
  return value;
}

high_board_t high_board_t::place(const int bit) const {
  const auto move = side_t(1) << bit;
  NON_WASM_ASSERT(!middle() && (move & empty_mask()));
  side_t after[2] = {side(0), side(1)};
  after[turn()] |= move;
  return high_board_t(after[0], after[1], s.ply_ + 1);
}

high_board_t high_board_t::place(const int x, const int y) const {
  NON_WASM_ASSERT(unsigned() < 6 && unsigned(y) < 6);
  return place(3*(x%3) + y%3 + 16*(2*(x/3) + y/3));
}

high_board_t high_board_t::rotate(const int q, const int d) const {
  NON_WASM_ASSERT(middle() && 0 <= q && q < 4 && d==1 || d==-1);
  side_t after[2] = {side(0), side(1)};
  WASM_NOUNROLL
  for (const int s : range(2)) {
    const auto old = quadrant(after[s], q);
    quadrant_t quad = 0;
    for (const int i : range(9)) {
      if (old >> i & 1) {
        const int x = i / 3;
        const int y = i % 3;
        quad |= 1 << (4-d*(3*y-x-2));
      }
    }
    after[s] ^= side_t(old ^ quad) << 16*q;
  }
  return high_board_t(after[0], after[1], s.ply_ + 1);
}

board_t high_board_t::board() const {
  board_t board = 0;
  WASM_NOUNROLL
  for (const int q : range(4)) {
    uint16_t quad = 0;
    WASM_NOUNROLL
    for (int i = 8; i >= 0; i--) {
      const int j = 16*q + i;
      quad = 3*quad + (s.side_[0] >> j & 1) + 2*(s.side_[1] >> j & 1);
    }
    board |= board_t(quad) << 16*q;
  }
  return board;
}

high_board_t high_board_t::from_board(const board_t board, const bool middle) {
  side_t side0 = 0, side1 = 0;
  for (const int q : range(4)) {
    uint16_t quad = board >> 16*q;
    for (const int i : range(9)) {
      const int s = quad % 3;
      quad /= 3;
      const int j = 16*q + i;
      side0 |= side_t(s == 1) << j;
      side1 |= side_t(s == 2) << j;
    }
  }
  return high_board_t(side0, side1, 2*popcount(side0 | side1) - middle);
}

#ifndef __wasm__
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

string high_board_t::name() const {
  return tfm::format("%d%s", board(), middle() ? "m" : "");
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
