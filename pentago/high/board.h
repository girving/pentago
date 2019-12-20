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
#include "pentago/data/block_cache.h"
#include "pentago/utility/array.h"
namespace pentago {

struct high_board_t {
  const board_t board; // Board state: 0 for black (first player), 1 for white (second player)
  const int turn; // Whose turn is it: 0 (black) or 1 (white)
  const bool middle; // Did we already place a stone?  I.e., are we halfway through the move?
  const Array<const int,2> grid; // x-y major, 0-0 is lower left, values are 0-empty, 1-black, 2-white

  high_board_t(const board_t board, const bool middle);
  ~high_board_t();

  // Total number of stones
  int count() const;

  // Is the game over?
  bool done() const;

  // Moves which follow this one.  Note that high level moves are "half" of a regular move:
  // there is one move to place a stone and one move to rotate it.
  vector<high_board_t> moves() const;

  // Place a stone at the given location
  high_board_t place(const int x, const int y) const;

  // Rotate the given quadrant in the given direction (-1 or 1)
  high_board_t rotate(const int qx, const int qy, const int d) const;

  // value() assuming done()
  int immediate_value() const;

  // 1 if the player to move wins, 0 for tie, -1 if the player to move loses
  int value(const block_cache_t& cache) const;

  // Same as value, but verify consistency with minimum depth tree search.
  int value_check(const block_cache_t& cache) const;

  // Compare against a bunch of samples and return loss,tie,win counts
  static Vector<int,3> sample_check(const block_cache_t& cache, RawArray<const board_t> boards,
                                    RawArray<const Vector<super_t,2>> wins);

  bool operator==(const high_board_t& other) const {
    return board==other.board && middle==other.middle;
  }

  string name() const;
  friend ostream& operator<<(ostream& output, const high_board_t& board);
  static high_board_t parse(const string& name);
  template<class T> friend struct std::hash;
};

}  // namespace pentago
namespace std {
template<> struct hash<pentago::high_board_t> {
  size_t operator()(const pentago::high_board_t& b) const {
    return hash<pentago::board_t>()(b.middle ? ~b.board : b.board);
  }
};
}  // namespace std
