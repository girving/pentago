// Core tree search engine

#include <pentago/board.h>
#include <pentago/score.h>
#include <pentago/moves.h>
#include <pentago/stat.h>
#include <pentago/table.h>
#include <pentago/utility/sort.h>
#include <other/core/math/min.h>
#include <other/core/python/wrap.h>
#include <other/core/utility/interrupts.h>
namespace pentago {

using std::max;
using std::cout;
using std::endl;
using std::swap;
using namespace other;

namespace {

// Evaluate position based only on current state and transposition table
inline score_t quick_evaluate(board_t board) {
  // Is the current position a win, loss, or immediate tie?
  const int st = status(board);
  if (st)
    return exact_score(1+(st&1)-(st>>1));
  // Did we already compute this value?
  return lookup(board);
}

// Same as evaluate, but assume board status and transposition table have supplied no information.
score_t evaluate_recurse(int depth, board_t board) {
  STAT(total_expanded_nodes++);
  STAT(expanded_nodes[depth]++);
  // Collect possible moves
  // board_t moves[total] = {...};
  MOVES(board)

  // Check status and transposition table for each move
  score_t best = score(depth,0);
  for (int i=total-1;i>=0;i--) {
    score_t sc = lift(quick_evaluate(moves[i]));
    if (sc>>2 >= depth)
      switch (sc&3) {
        case 2: return sc; // Win!
        case 1: best = sc; // Tie: falls through to next case
        case 0: swap(moves[i],moves[--total]); // Loss or tie: remove from the list of moves
      }
  }

  // If we're out of moves, we're done
  if (!total)
    goto done;

  // Are we out of recursion depth?
  if (depth==1) {
    best = score(1,1);
    goto done;
  }

  // No move ordering for now, since all the remaining moves are ties as far the transposition table knows
  for (int i=0;i<total;i++) {
    score_t sc = lift(evaluate_recurse(depth-1,moves[i]));
    if ((sc&3)==2) {
      best = sc;
      goto done;
    } else if ((best&3)<(sc&3))
      best = sc;
  }

  // Store results and finish
  done:
  store(board,best);
  check_interrupts();
  return best;
}

// Evaluate a position down to a certain game tree depth.  If the depth is exceeded, assume a tie.
// We maintain the invariant that player 0 is always the player to move.  The result is 1 for a win
// 0 for tie, -1 for loss.
score_t evaluate(int depth, board_t board) {
  // We can afford error detection here since recursion happens into a different function
  OTHER_ASSERT(depth>=0);
  check_board(board);
  set_table_type(normal_table);

  // We don't need to search past the end of the game
  depth = min(depth,36-popcount(unpack(board,0)|unpack(board,1)));

  // Exit immediately if possible
  score_t sc = quick_evaluate(board);
  if (sc>>2 >= depth)
    return sc;

  // Otherwise, recurse into children
  return evaluate_recurse(depth,board);
}

// Evaluate position based only on current state and transposition table, ignoring rotations or white five-in-a-row
template<bool black> inline score_t simple_quick_evaluate(side_t side0, side_t side1) {
  // If we're white, check if we've lost
  if (!black && rotated_won(side1))
    return exact_score(0);
  // Did we already compute this value?
  return lookup(pack(side0,side1));
}

// Evaluate position ignoring rotations and white five-in-a-row
template<bool black> score_t simple_evaluate_recurse(int depth, side_t side0, side_t side1) {
  STAT(total_expanded_nodes++);
  STAT(expanded_nodes[depth]++);

  // Collect possible moves
  SIMPLE_MOVES(side0,side1)

  // Check status and transposition table for each move
  score_t best = black?score(depth,1):exact_score(0);
  for (int i=total-1;i>=0;i--) {
    score_t sc = lift(simple_quick_evaluate<!black>(side1,moves[i]));
    if (sc>>2 >= depth) {
      if ((sc&3)>=(black?2:1))
        return sc; // Win for black, or tie for white
      else // Tie for black, or loss for white: remove from the list of moves
        swap(moves[i],moves[--total]);
    }
  }

  // If we're out of moves, we're done
  if (!total)
    goto done;

  // Are we out of recursion depth?
  if (depth==1) {
    best = score(1,1);
    goto done;
  }

  {
    // Compute how close we are to a black win
    int order[total]; // We'll sort moves in ascending order of this array
    for (int i=total-1;i>=0;i--) {
      int closeness = black?rotated_win_closeness(moves[i],side1)
                           :rotated_win_closeness(side1,moves[i]);
      int distance = 6-(closeness>>16);
      if (distance==6 || distance>(depth-black)/2) { // We can't reach a black win within the given search depth
        if (!black) {
          STAT(distance_prunes++);
          best = score(distance==6?36:2*distance-1+black,1);
          goto done;
        } else {
          swap(moves[i],moves[total-1]);
          swap(order[i],order[--total]);
        }
      }
      order[i] = black?-closeness:closeness;
    }

    // Sort moves based on order
    insertion_sort(total,order,moves);

    // Optionally print out move ordering information
    if (0) {
      cout << (black?"order black =":"order white =");
      for (int i=0;i<total;i++) {
        int o = black?-order[i]:order[i];
        cout << ' '<<(6-(o>>16))<<','<<(o&0xffff);
      }
      cout<<endl;
    }
  }

  // Recurse
  for (int i=0;i<total;i++) {
    score_t sc = lift(simple_evaluate_recurse<!black>(depth-1,side1,moves[i]));
    if ((sc&3)>=(black?2:1)) {
      best = sc;
      goto done;
    }
  }

  // Store results and finish
  done:
  store(pack(side0,side1),best);
  check_interrupts();
  return best;
}

// Evaluate a position down to a certain game tree depth, ignoring rotations and white wins.
score_t simple_evaluate(int depth, board_t board) {
  // We can afford error detection here since recursion happens into a different function
  OTHER_ASSERT(depth>=0);
  check_board(board);
  set_table_type(simple_table);

  // Determine whether player 0 is black (first to move) or white (second to move)
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  bool black = black_to_move(board);

  // We don't need to search past the end of the game
  depth = min(depth,36-popcount(side0|side1));

  // Exit immediately if possible
  score_t sc = black?simple_quick_evaluate<true>(side0,side1)
                    :simple_quick_evaluate<false>(side0,side1);
  if (sc>>2 >= depth)
    return sc;

  // Otherwise, recurse into children
  return black?simple_evaluate_recurse<true >(depth,side0,side1)
              :simple_evaluate_recurse<false>(depth,side0,side1);
}

}
}
using namespace pentago;
using namespace other::python;

void wrap_engine() {
  OTHER_FUNCTION(evaluate)
  OTHER_FUNCTION(simple_evaluate)
}
