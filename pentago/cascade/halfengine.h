// Rotation abstracted forward search over halfsuper_t
//
// A port of the value search in pentago/search/superengine.h to halfsuper_t, built to compile
// freestanding for wasm.  The superengine evaluates a board for all 256 local rotation states at
// once, but each node only ever needs the 128 states of one parity, since moving to a child applies
// exactly one quarter turn (rmax flips parity).  So every set here is a halfsuper_t of the node's
// parity: the root has parity 0 (no rotation), and parity alternates with depth.
//
// As in the superengine, one player is "aggressive" and loses ties, so a search is a window of size
// two and each state needs one bit.  Differences from the superengine:
//
//   1. The transposition table is owned by the engine, not global, and its 40 byte entries are
//      keyed by the raw board, parity, and aggressor.  There is no symmetry standardization: all
//      positions in one search share the root's frame, so the only transpositions lost are between
//      boards related by a symmetry, which are rare away from the opening.
//   2. Move ordering and distance pruning use the same arbitrarily rotated win distances, but the
//      1.2 MB table is computed at startup (tens of milliseconds) instead of shipped.
//   3. No block cache, move limit, debug mode, tracing, statistics, or child move reporting.
#pragma once

#include "pentago/base/board.h"
#include "pentago/mid/halfsuper.h"
namespace pentago {

class tie_prover_t;

// Tie prover cutoffs inside the search.  At a node with at least min_depth plies left and at most
// max_states undecided rotation states, ask the tie prover about each state: if the defender holds
// at an aggressive node, the aggressor can't win; if the player to move holds at a passive node, it
// can't lose.  Both are facts about the whole game, so they hold at any depth.
struct halfengine_cutoffs_t {
  tie_prover_t* prover = 0;  // Null to disable
  int min_depth = 6;
  int max_states = 64;
  int depth = 0;             // Free replies for the tie prover (the cascade uses 1)
  double seconds = 0.01;     // Time limit per tie prover call
};

struct halfinfo_t {
  halfsuper_t known, wins;  // For the player to move; a "win" for the passive player includes ties
};

class halfengine_t {
 public:
  // The table has 2^bits entries of 40 bytes each, and is allocated once
  explicit halfengine_t(const int bits);
  ~halfengine_t();
  halfengine_t(const halfengine_t&) = delete;
  void operator=(const halfengine_t&) = delete;

  // Search to the given depth with side0 to move and no rotation applied.  If aggressive, returns
  // whether side0 wins within depth plies; otherwise, whether side0 avoids losing within depth plies.
  bool evaluate(const bool aggressive, const int depth, const side_t side0, const side_t side1);

  // Iterative deepening: 1 if side0 (to move) wins, -1 if it loses, 0 for a tie, 2 if unknown after
  // the depth that crosses the time limit.  clock returns seconds.  Wins for the player to move are
  // sought at odd depths and losses at even depths, since a search to depth d covers all shorter
  // lines; at full depth both searches are needed to see a tie.
  int search(const side_t side0, const side_t side1, const double seconds, double (*clock)());

  // Clear the table
  void clear();

  halfengine_cutoffs_t cutoffs;
  uint64_t cutoff_calls = 0, cutoff_hits = 0;  // Tie prover calls, and those that settled a state

  // For tests: side as it stands in rotation state i of a halfsuper_t of the given parity
  static side_t rotate_state(const side_t side, const bool parity, const int i);

  // For tests: the superengine's arbitrarily_rotated_win_closeness, computed from runtime tables
  static int closeness(const side_t black, const side_t white);

  // For tests: halfsuper_wins, computed from runtime tables
  static halfsuper_t fast_wins(const side_t side, const bool parity);

  // Internal (public for the implementation's helper templates)
  struct entry_t;
  struct node_t;
  template<bool aggressive> node_t shallow(const int depth, const bool parity, const side_t side0,
                                           const side_t side1, const halfsuper_t wins0,
                                           const halfsuper_t wins1, const halfsuper_t interesting);
  template<bool aggressive> halfinfo_t recurse(const int depth, const bool parity, const side_t side0,
                                               const side_t side1, node_t node, const halfsuper_t important);
  template<bool aggressive> void store(const int depth, const uint64_t hash, const halfinfo_t& info);
 private:
  const int bits;
  entry_t* table;
};

}  // namespace pentago
