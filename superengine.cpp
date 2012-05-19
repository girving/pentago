// Core tree search engine abstracted over rotations

#include "superengine.h"
#include "score.h"
#include "sort.h"
#include "stat.h"
#include "moves.h"
#include "superscore.h"
#include "supertable.h"
#include "trace.h"
#include <other/core/python/module.h>
#include <other/core/python/stl.h>
#include <other/core/structure/HashtableIterator.h>
#include <other/core/structure/Tuple.h>
#include <other/core/utility/Hasher.h>
#include <other/core/utility/interrupts.h>
namespace pentago {

using std::max;
using std::cout;
using std::endl;
using std::swap;
using namespace other;

// Limit blacks options to the given number of moves (chosen after move ordering)
static int super_move_limit = 36;

void set_super_move_limit(int limit) {
  OTHER_ASSERT(limit>0);
  super_move_limit = limit;
}

// Turn on debug mode
static bool debug = false;

void set_super_debug(bool on) {
  debug = on;
}

// Evaluate everything we can about a position without recursing into children.
template<bool black,bool debug> static inline superdata_t OTHER_ALWAYS_INLINE super_shallow_evaluate(const int depth, const side_t side0, const side_t side1, const super_t wins0, const super_t wins1, const super_t interesting) {
  // Check whether the current state is a win for either player
  superinfo_t info;
  info.known = wins0|wins1;
  info.wins = black?wins0&~wins1:wins0;

  // If zero moves remain, white wins
  if (!depth) {
    if (!black)
      info.wins |= ~info.known;
    info.known = ~super_t(0);
    goto done;
  }

  // If we're white, and only one move remains, any safe move is a win
  if (!black && depth<=1) {
    const super_t safe = rmax(~wins1);
    info.wins |= safe&~info.known;
    info.known |= safe;
  }

  // Exit early if possible
  if (!(interesting&~info.known))
    goto done;

  // Look up the position in the transposition table
  {
    superdata_t data;
    data.lookup = super_lookup<black>(depth,side0,side1);
    superinfo_t& info2 = data.lookup.info;
    OTHER_ASSERT(!((info.wins^info2.wins)&info.known&info2.known));
    if (debug) {
      // In engine debug mode, we write to the transposition table but read only to check whether different computations match
      info2 = info;
    } else {
      info2.wins = info.wins|(info2.wins&~info.known);
      info2.known |= info.known;
    }
    data.wins1 = wins1;
    return data;
  }

  done:
  superdata_t data;
  data.lookup.hash = 0;
  data.lookup.info = info;
  data.wins1 = wins1;
  return data;
}

// Use only from top level
superdata_t super_shallow_evaluate(const int depth, const side_t side0, const side_t side1, const super_t interesting) {
  const bool black = black_to_move(pack(side0,side1));
  auto shallow = black?debug?super_shallow_evaluate<true, true>:super_shallow_evaluate<true, false>
                      :debug?super_shallow_evaluate<false,true>:super_shallow_evaluate<false,false>;
  return shallow(depth,side0,side1,super_wins(side0),super_wins(side1),interesting);
}

template<> struct results_t<true> {
  typedef results_t type;
  Array<side_t> moves; // list of moves we know about
  Hashtable<side_t,super_t> immediate_wins; // moves where we when without needing to rotate
  Hashtable<side_t,superinfo_t> children; // information about children (note that we lose if the child wins)
  void immediate_win(side_t move, super_t immediate) { immediate_wins.set(move,immediate); if (children.set(move,0)) moves.append(move); }
  void child(side_t move, superinfo_t info) { if (children.set(move,info)) moves.append(move); }
  results_t return_(superinfo_t info) const { return *this; }
};

// Evaluate a position for all important rotations, returning the set of rotations in which we win.
// Note that a "win" for white includes ties.  The returned set of known rotations definitely includes
// all important rotations, but may include others as well.
template<bool remember,bool black,bool debug> static typename results_t<remember>::type super_evaluate_recurse(const int depth, const side_t side0, const side_t side1, superdata_t data, const super_t important) {
  STAT(total_expanded_nodes++);
  STAT(expanded_nodes[depth]++);
  PRINT_STATS(24);
  superinfo_t& info = data.lookup.info;
  super_t possible = 0; // Keep track of possible wins that we might have missed
  results_t<remember> results; // Optionally keep track of results about children 

  // Consistency check
  OTHER_ASSERT(info.valid());

  // Be verbose if desired
  TRACE_VERBOSE_START(depth,pack(side0,side1));
  TRACE_VERBOSE("input %s %s",subset(info.known,verbose),subset(info.wins,verbose));

  // Collect possible moves
  SIMPLE_MOVES(side0,side1)
  if (remember && !(side0|side1)) { // Handle starting moves specially
    // After taking rotations into account, there are only three unique starting moves
    total = 3;
    moves[0] = 1<<(3*1+1); // center
    moves[1] = 1<<(3*1+2); // side
    moves[2] = 1<<(3*0+2); // corner
  }

  // Do a shallow evaluation of each move
  const super_t theirs = data.wins1;
  superdata_t* children = (superdata_t*)alloca(total*sizeof(superdata_t));
  for (int i=total-1;i>=0;i--) {
    const side_t move = moves[i];
    // Do we win without a rotation?  If we're white, it's safe to wait until after the rotation to check.
    const super_t ours = super_wins(move);
    if (black) {
      super_t immediate = ours&~theirs;
      info.wins |= immediate&~info.known;
      info.known |= immediate;
      results.immediate_win(move,immediate);
    }

    // Do a shallow evaluation of the child position
    const super_t mask = rmax(important&~info.known);
    children[i] = super_shallow_evaluate<!black,debug>(depth-1,side1,move,theirs,ours,mask);
    const superinfo_t& child = children[i].lookup.info;
    TRACE(trace_dependency(depth,pack(side0,side1),depth-1,pack(side1,move),child));
    const super_t wins = rmax(~child.wins&child.known);
    info.wins |= wins&~info.known;
    info.known |= wins;
    results.child(move,child);

    // Exit early if possible
    if (!(important&~info.known)) {
      TRACE_VERBOSE("early exit, move %lld, result %s",move,subset(info.wins,verbose));
      goto done;
    }

    // Discard the move if we already know enough about it
    if (!(mask&~child.known)) {
      possible |= rmax(~child.known);
      swap(moves[i],moves[total-1]);
      swap(children[i],children[total-1]);
      total--;
    }
  }

  {
    // Check how close we are to a black win, pretending that black gets to choose the rotation arbitrarily
    int order[total]; // We'll sort moves in ascending order of this array
    for (int i=total-1;i>=0;i--) {
      int closeness = black?arbitrarily_rotated_win_closeness(moves[i],side1)
                           :arbitrarily_rotated_win_closeness(side1,moves[i]);
      if (!remember) {
        int distance = 6-(closeness>>16);
        if (distance==6 || distance>(depth-black)/2) { // We can't reach a black win within the given search depth
          if (!black) {
            STAT(distance_prunes++);
            info.wins = info.known = ~super_t(0);
            TRACE_VERBOSE("distance prune, depth %d, distance %d",depth,distance);
            goto done;
          } else {
            swap(moves[i],moves[total-1]);
            swap(children[i],children[total-1]);
            swap(order[i],order[total-1]);
            total--;
          }
        }
      }
      order[i] = black?-closeness:closeness;
    }

    // Are we out of recursion depth?
    if (depth==1 && total) {
      TRACE_VERBOSE("out of recursion depth");
      goto done;
    }

    // Sort moves and children arrays based on order
    insertion_sort(total,order,moves,children);

    // Optionally limit black's choices
    if (black)
      total = min(total,super_move_limit);

    // Recurse
    for (int i=0;i<total;i++) {
      const side_t move = moves[i];
      super_t mask = rmax(important&~info.known);
      superinfo_t child = super_evaluate_recurse<false,!black,debug>(depth-1,side1,move,children[i],mask);
      super_t wins = rmax(~child.wins&child.known);
      info.wins |= wins&~info.known;
      info.known |= wins;
      results.child(move,child);
      if (!(important&~info.known)) {
        TRACE_VERBOSE("pruned, move %lld, result %s",move,subset(info.wins,verbose));
        goto done;
      }
      possible |= rmax(~child.known);
    }

    // If we've analyzed all children, any move that isn't a possible win is a loss
    TRACE_VERBOSE("analyzed all children: wins %s, possible %s",subset(info.wins,verbose),subset(possible,verbose));
    info.known |= ~possible;
  }

  // Store results and finish
  done:
  TRACE_VERBOSE("storing depth %d, board %lld, result %s",depth,superstandardize(pack(side0,side1)).x,subset(info.wins,verbose));
  TRACE(trace_check(depth,pack(side0,side1),info,"super_evaluate_recurse"));
  if (data.lookup.hash)
    super_store<black>(depth,data.lookup);
  check_interrupts();
  return results.return_(info);
}

// Use only from top level
template<bool remember> typename results_t<remember>::type super_evaluate_recurse(const int depth, const side_t side0, const side_t side1, superdata_t data, const super_t important) {
  const bool black = black_to_move(pack(side0,side1));
  #define CASE(b,d) if (black==b && debug==d) return super_evaluate_recurse<remember,b,d>(depth,side0,side1,data,important);
  CASE(0,0) CASE(0,1) CASE(1,0) CASE(1,1)
  OTHER_UNREACHABLE();
}

static inline score_t to_score(bool black, int depth, bool win) {
  int value = black+win;
  return value==1?score(depth,value):exact_score(value);
}

// Driver for evaluation abstracted over rotations
score_t super_evaluate(int depth, const board_t board, const Vector<int,4> rotation) {
  // We can afford error detection here since recursion happens into a different function
  OTHER_ASSERT(supertable_bits()>=10);
  OTHER_ASSERT(depth>=0);
  check_board(board);

  // Determine whether player 0 is black (first to move) or white (second to move)
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const bool black = black_to_move(board);

  // We don't need to search past the end of the game
  depth = min(depth,36-popcount(side0|side1));

  // Exit immediately if possible
  const super_t start = super_t::singleton(rotation);
  const superdata_t data = super_shallow_evaluate(depth,side0,side1,start);
  if (data.lookup.info.known(rotation))
    return to_score(black,depth,data.lookup.info.wins(rotation));
  if (!depth)
    return score(0,1);

  // Otherwise, recurse into children
  const superinfo_t info = super_evaluate_recurse<false>(depth,side0,side1,data,start);
  OTHER_ASSERT(info.known(rotation));
  return to_score(black,depth,info.wins(rotation));
}

// Evaluate enough children to determine who wins, and return the results
vector<Tuple<rotated_board_t,score_t>> super_evaluate_children(const int depth, const board_t board, const Vector<int,4> rotation) {
  // We can afford error detection here since recursion happens into a different function
  OTHER_ASSERT(supertable_bits()>=10);
  OTHER_ASSERT(depth>=1);
  check_board(board);

  // Determine whether player 0 is black (first to move) or white (second to move)
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const bool black = black_to_move(board);

  // We don't need to search past the end of the game
  const int infinite_depth = 36-popcount(side0|side1),
            search_depth = min(depth,infinite_depth),
            report_depth = depth>=infinite_depth?36:depth;

  // If we already know the value of this position, super_evaluate_recurse won't give us any child information.
  // Therefore, pretend we don't know.
  const super_t start = super_t::singleton(rotation);
  superdata_t data = super_shallow_evaluate(search_depth,side0,side1,start);
  data.lookup.info.wins &= ~start;
  data.lookup.info.known &= ~start;

  // Recurse into children
  const results_t<true> results = super_evaluate_recurse<true>(search_depth,side0,side1,data,start);
  OTHER_ASSERT(results.moves.size());

  // Analyze results
  vector<Tuple<rotated_board_t,score_t>> children;
  for (const side_t move : results.moves) {
    const board_t after = pack(side1,move);
    // If there's an immediate win, generate a move with no rotation.  Such moves are legal only in this case.
    if (results.immediate_wins.get_default(move,super_t(0))(rotation))
      children.push_back(tuple(tuple(after,as_tuple(rotation)),to_score(black,report_depth,true)));
    else {
      // Otherwise, report results for all known children
      const superinfo_t& info = results.children.get(move);
      for (Vector<int,4> r : single_rotations)
        if (info.known(rotation+r)) {
          children.push_back(tuple(tuple(after,as_tuple(rotation+r)),to_score(black,report_depth,!info.wins(rotation+r))));
        }
    }
  }
  OTHER_ASSERT(children.size());
  return children;
}

}
using namespace pentago;
using namespace other::python;

void wrap_superengine() {
  OTHER_FUNCTION(super_evaluate)
  OTHER_FUNCTION(super_evaluate_children)
  OTHER_FUNCTION(set_super_move_limit)
  OTHER_FUNCTION(set_super_debug)
}
