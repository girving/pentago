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

#include "pentago/base/board.h"
#ifndef __wasm__
#include "pentago/data/block_cache.h"
#endif
NAMESPACE_PENTAGO

class high_board_t {
  uint64_t side_[2];  // black, white (first player, second player)
  uint32_t ply_;

  high_board_t(const side_t side0, const side_t side1, const int ply) : side_{side0, side1}, ply_(ply) {}
public:

  high_board_t() : side_{0, 0}, ply_(0) {}

  side_t side(const int s) const {
    assert(unsigned(s) < 2); return side_[s];
  }

  Vector<side_t,2> sides() const { return vec(side_[0], side_[1]); }

  // Number of moves from the start of the game, counting stone placement and rotation separately.
  int ply() const { return ply_; }

  // Did we already place a stone?  I.e., are we halfway through the move?
  bool middle() const { return ply_ & 1; }

  bool operator==(const high_board_t other) const {
    return side_[0] == other.side_[0] && side_[1] == other.side_[1] && ply_ == other.ply_;
  }

  // Total number of stones
  int count() const { return (ply_ + 1) >> 1; }

  // Whose turn is it: 0 (black) or 1 (white)
  int turn() const { return (ply_ >> 1) & 1; }

  // Is the game over?
  bool done() const;

  // Place a stone at the given location
  high_board_t place(const int bit) const;
  high_board_t place(const int x, const int y) const;

  // Rotate the given quadrant in the given direction (-1 or 1)
  high_board_t rotate(const int q, const int d) const;

  // value() assuming done()
  int immediate_value() const;

  side_t empty_mask() const { return side_mask ^ side_[0] ^ side_[1]; }

#ifndef __wasm__
  static high_board_t from_board(const board_t board, const bool middle);
  board_t board() const { return pack(side_[0], side_[1]); }

  // Moves which follow this one.  Note that high level moves are "half" of a regular move:
  // there is one move to place a stone and one move to rotate it.
  Array<const high_board_t> moves() const;

  // 1 if the player to move wins, 0 for tie, -1 if the player to move loses
  int value(const block_cache_t& cache) const;

  // Same as value, but verify consistency with minimum depth tree search.
  int value_check(const block_cache_t& cache) const;

  // Compare against a bunch of samples and return loss,tie,win counts
  static Vector<int,3> sample_check(const block_cache_t& cache, RawArray<const board_t> boards,
                                    RawArray<const Vector<super_t,2>> wins);
  string name() const;
  friend ostream& operator<<(ostream& output, const high_board_t board);
  static high_board_t parse(const string& name);
  template<class T> friend struct std::hash;
#endif  // !__wasm__
};

END_NAMESPACE_PENTAGO
#ifndef __wasm__
namespace std {
template<> struct hash<pentago::high_board_t> {
  size_t operator()(const pentago::high_board_t board) const {
    auto b = board.board();
    if (board.middle()) b = ~b;
    return hash<uint64_t>()(b);
  }
};
}  // namespace std
#endif  // !__wasm__
