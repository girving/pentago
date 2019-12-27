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
#include "pentago/utility/pile.h"
#ifndef __wasm__
#include "pentago/data/block_cache.h"
#endif
namespace pentago {

class high_board_t {
  // Split into two uint32_t's to allow 4-byte alignment
  uint32_t rep_[2];  // middle ? ~board : board
  uint64_t rep() const { return rep_[0] | uint64_t(rep_[1]) << 32; }
public:

  high_board_t() : rep_{0, 0} {}
  high_board_t(const board_t board, const bool middle);
  ~high_board_t();

  // Board state: 0 for black (first player), 1 for white (second player)
  board_t board() const { const auto d = rep(); return middle() ? ~d : d; }

  // Did we already place a stone?  I.e., are we halfway through the move?
  bool middle() const { return bool(rep_[1] & 1<<31); }

  bool operator==(const high_board_t other) const { return rep() == other.rep(); }

  // Total number of stones
  int count() const;

  // Is the game over?
  bool done() const;

  // Whose turn is it: 0 (black) or 1 (white)
  int turn() const;

  // Moves which follow this one.  Note that high level moves are "half" of a regular move:
  // there is one move to place a stone and one move to rotate it.
  pile<high_board_t,36> moves() const;

  // Place a stone at the given location
  high_board_t place(const int x, const int y) const;

  // Rotate the given quadrant in the given direction (-1 or 1)
  high_board_t rotate(const int qx, const int qy, const int d) const;

  // value() assuming done()
  int immediate_value() const;

#ifndef __wasm__
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

}  // namespace pentago
#ifndef __wasm__
namespace std {
template<> struct hash<pentago::high_board_t> {
  size_t operator()(const pentago::high_board_t board) const {
    return hash<uint64_t>()(board.rep());
  }
};
}  // namespace std
#endif  // !__wasm__
