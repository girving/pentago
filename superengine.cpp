// Core tree search engine abstracted over rotations

#include "score.h"
#include "stat.h"
#include "moves.h"
#include "superscore.h"
#include <other/core/python/module.h>
#include <other/core/python/stl.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/Hasher.h>
#include <other/core/utility/interrupts.h>
namespace pentago {

using std::max;
using std::cout;
using std::endl;
using std::swap;
using namespace other;

namespace {

// Evaluate a position for all important rotations, returning the set of rotations in which we win.
// Note that a "win" for white includes ties.  We maintain the invariant that no important rotation
// is an immediate win for either player.
template<bool black> super_t super_evaluate_recurse(int depth, side_t side0, side_t side1, super_t important) {
  STAT(expanded_nodes++);
  PRINT_STATS(24);

  // Pretend we automatically win for all unimportant moves
  super_t wins = ~important;

  // If we're white, and only one move remains, any immediate safe move is a win
  super_t theirs = super_wins(side1);
  if (!black && depth==1) {
    wins |= rmax(~theirs);
    if (!~wins)
      goto done;
  }

  {
    // Collect possible moves
    SIMPLE_MOVES(side0,side1)

    // If no moves remain, it's a tie
    if (!total) {
      wins = black?super_t(0):~super_t(0);
      goto done;
    }

    // Check immediate status for each move
    for (int i=0;i<total;i++) {
      super_t ours = super_wins(moves[i]);
      if (black)
        ours &= ~theirs;
      wins |= ours|rmax(ours); // We win if we win immediately or after one rotation
      if (!~wins) // If we always win, we're done
        goto done;
    }

    // Are we out of recursion depth?
    if (depth==1)
      goto done;

    // Recurse
    for (int i=0;i<total;i++) {
      super_t mask = rmax(~wins)&~theirs;
      super_t losses = super_evaluate_recurse<!black>(depth-1,side1,moves[i],mask);
      wins |= rmax(~losses&mask);
      if (!~wins)
        goto done;
    }
  }

  // Store results and finish
  done:
  check_interrupts();
  return wins;
}

// Driver for evaluation abstracted over rotations
score_t super_evaluate(int depth, board_t board, Vector<int,4> rotation) {
  // We can afford error detection here since recursion happens into a different function
  OTHER_ASSERT(depth>=0);
  check_board(board);

  // Determine whether player 0 is black (first to move) or white (second to move)
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const bool black = black_to_move(board);

  // Exit immediately if possible
  super_t start = super_t::singleton(rotation);
  bool ours = super_wins(side0)&start, theirs = super_wins(side1)&start;
  if (ours|theirs)
    return exact_score((black?ours&~theirs:ours)?2:1);
  if (!depth)
    return score(0,1);

  // Otherwise, recurse into children
  super_t wins = black?super_evaluate_recurse<true >(depth,side0,side1,start)
                      :super_evaluate_recurse<false>(depth,side0,side1,start);
  int value = black+wins(rotation);
  return value==1?score(depth,value):exact_score(value);
}

typedef Tuple<board_t,Tuple<int,int,int,int>> rotated_board_t;

// Evaluate enough children to determine who wins, and return the results
unordered_map<rotated_board_t,score_t,Hasher> super_evaluate_children(int depth, board_t board, Vector<int,4> rotation) {
  OTHER_ASSERT(depth>=1);
  check_board(board);

  // Generate standardized moves
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const bool black = black_to_move(board);
  set<side_t> standard_moves;
  if (!popcount(side1)) {
    // After taking rotations into account, there are only three unique starting moves
    standard_moves.insert(1<<(3*1+1)); // center
    standard_moves.insert(1<<(3*1+2)); // side
    standard_moves.insert(1<<(3*0+2)); // corner
  } else {
    SIMPLE_MOVES(side0,side1);
    OTHER_ASSERT(total);
    standard_moves.insert(moves,moves+total);
  }

  // Consider enough moves to determine who wins
  const super_t start = super_t::singleton(rotation),
                single = rmax(start),
                theirs = super_wins(side1);
  unordered_map<rotated_board_t,score_t,Hasher> results;
  for (side_t move : standard_moves) {
    super_t ours = super_wins(move);
    if (black)
      ours &= ~theirs;
    super_t wins = rmax(ours&start)|ours; // Determine which rotations win immediately
    if (!(wins&single)) {
      super_t mask = single&~theirs;
      super_t losses = black?depth>1?super_evaluate_recurse<false>(depth-1,side1,move,mask):~super_t(0)
                            :depth>1?super_evaluate_recurse<true >(depth-1,side1,move,mask): super_t(0);
      wins |= ~losses&mask;
    }
    bool won = false;
    for (Vector<int,4> r : single_rotations) {
      auto rb = tuple(pack(side1,move),as_tuple(rotation+r));
      won |= wins(rotation+r);
      int value = black+wins(rotation+r);
      results[rb] = value==1?score(depth,value):exact_score(value);
    }
    if (won)
      break;
  }
  return results;
}

}
}
using namespace pentago;
using namespace other::python;

void wrap_superengine() {
  OTHER_FUNCTION(super_evaluate)
  OTHER_FUNCTION(super_evaluate_children)
}
