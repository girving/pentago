// High level interface to board
//
// high_board_t wraps a nice class interface around the bare uint64_t that is board_t.
// It also splits the underlying place+rotation ply into two separate plys, which
// corresponds more closely to the final frontend interface to the game.
//
// When converted to a string, boards with the player to move about to place a stone
// are represented as a bare int; boards with the player to move about to rotate have
// a trailing 'm'.  For example, 1m.
#pragma once

#include "pentago/high/board_c.h"
#if PENTAGO_CPP
#include "pentago/base/board.h"
#include "pentago/utility/array.h"
#include "pentago/utility/vector.h"
#endif
NAMESPACE_PENTAGO

struct high_board_t {
  high_board_s s;

  high_board_t() { s.side_[0] = s.side_[1] = s.ply_ = 0; }
  high_board_t(const high_board_s s) : s(s) {}
  high_board_t(const side_t side0, const side_t side1, const int ply) {
    s.side_[0] = side0; s.side_[1] = side1; s.ply_ = ply;
  }

  side_t side(const int i) const {
    assert(unsigned(i) < 2); return s.side_[i];
  }

  // Number of moves from the start of the game, counting stone placement and rotation separately.
  int ply() const { return s.ply_; }

  // Did we already place a stone?  I.e., are we halfway through the move?
  bool middle() const { return s.ply_ & 1; }

  bool operator==(const high_board_t other) const {
    return s.side_[0] == other.s.side_[0] && s.side_[1] == other.s.side_[1] && s.ply_ == other.s.ply_;
  }

  // Total number of stones
  int count() const { return (s.ply_ + 1) >> 1; }

  // Whose turn is it: 0 (black) or 1 (white)
  int turn() const { return (s.ply_ >> 1) & 1; }

#if PENTAGO_CPP
  Vector<side_t,2> sides() const { return vec(s.side_[0], s.side_[1]); }

  // Is the game over?  If so, what's the value()
  std::tuple<bool,int> done_and_value() const;
#endif

  // Is the game over?
  bool done() const;

  // value() assuming done()
  int immediate_value() const;

  // Place a stone at the given location
  high_board_t place(const int bit) const;
  high_board_t place(const int x, const int y) const;

  // Rotate the given quadrant in the given direction (-1 or 1)
  high_board_t rotate(const int q, const int d) const;

  side_t empty_mask() const { return side_mask ^ s.side_[0] ^ s.side_[1]; }

#if PENTAGO_CPP && !defined(__wasm__)
  static high_board_t from_board(const board_t board, const bool middle);
  board_t board() const { return pack(s.side_[0], s.side_[1]); }

  // Moves which follow this one.  Note that high level moves are "half" of a regular move:
  // there is one move to place a stone and one move to rotate it.
  Array<const high_board_t> moves() const;

  string name() const;
  friend ostream& operator<<(ostream& output, const high_board_t board);
  static high_board_t parse(const string& name);
  template<class T> friend struct std::hash;
#endif  // PENTAGO_CPP && !__wasm__
};

END_NAMESPACE_PENTAGO
#if PENTAGO_CPP && !defined(__wasm__)
namespace std {
template<> struct hash<pentago::high_board_t> {
  size_t operator()(const pentago::high_board_t board) const {
    auto b = board.board();
    if (board.middle()) b = ~b;
    return hash<uint64_t>()(b);
  }
};
}  // namespace std
#endif  // PENTAGO_CPP && !__wasm__
