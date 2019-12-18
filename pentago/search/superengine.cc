// Core tree search engine abstracted over rotations

#include "pentago/search/superengine.h"
#include "pentago/base/score.h"
#include "pentago/search/stat.h"
#include "pentago/base/moves.h"
#include "pentago/base/superscore.h"
#include "pentago/search/supertable.h"
#include "pentago/search/trace.h"
#include "pentago/data/block_cache.h"
#include "pentago/utility/sort.h"
#include <unordered_map>
namespace pentago {

using std::min;
using std::swap;
using std::make_pair;
using std::make_tuple;
using std::unordered_map;

// Limit black's options to the given number of moves (chosen after move ordering)
static int super_move_limit = 36;

void set_super_move_limit(int limit) {
  GEODE_ASSERT(limit>0);
  super_move_limit = limit;
}

// Turn on debug mode
static bool debug = false;

void set_super_debug(bool on) {
  debug = on;
}

// Block cache
static shared_ptr<const block_cache_t> block_cache;

void set_block_cache(shared_ptr<const block_cache_t> cache) {
  block_cache = cache;
}

// Evaluate everything we can about a position without recursing into children.
template<bool aggressive,bool debug> static inline superdata_t __attribute__((always_inline))
super_shallow_evaluate(const int depth, const side_t side0, const side_t side1, const super_t wins0,
                       const super_t wins1, const super_t interesting) {
  // Check whether the current state is a win for either player
  superinfo_t info;
  info.known = wins0|wins1;
  info.wins = aggressive?wins0&~wins1:wins0;

  // If zero moves remain, white wins
  if (!depth) {
    if (!aggressive)
      info.wins |= ~info.known;
    info.known = ~super_t(0);
    goto done;
  }

  // If we're white, and only one move remains, any safe move is a win
  if (!aggressive && depth<=1) {
    const super_t safe = rmax(~wins1);
    info.wins |= safe&~info.known;
    info.known |= safe;
  }

  // Exit early if possible
  if (!(interesting&~info.known))
    goto done;

  // If we have a block cache of precomputed "endgame" information, look there
  if (block_cache && block_cache->lookup(aggressive,side0,side1,info.wins)) {
    info.known = ~super_t(0);
    goto done;
  }

  // Look up the position in the transposition table
  {
    superdata_t data;
    data.lookup = super_lookup<aggressive>(depth,side0,side1);
    superinfo_t& info2 = data.lookup.info;
    GEODE_ASSERT(!((info.wins^info2.wins)&info.known&info2.known));
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
superdata_t super_shallow_evaluate(const bool aggressive, const int depth, const side_t side0, const side_t side1, const super_t interesting) {
  auto shallow = aggressive?debug?super_shallow_evaluate<true, true>:super_shallow_evaluate<true, false>
                           :debug?super_shallow_evaluate<false,true>:super_shallow_evaluate<false,false>;
  return shallow(depth,side0,side1,super_wins(side0),super_wins(side1),interesting);
}

template<> struct results_t<true> {
  typedef results_t type;
  vector<side_t> moves; // list of moves we know about
  unordered_map<side_t,super_t> immediate_wins; // moves where we when without needing to rotate
  unordered_map<side_t,superinfo_t> children; // information about children (note that we lose if the child wins)

  void immediate_win(side_t move, super_t immediate) {
    immediate_wins[move] = immediate;
    if (children.insert(make_pair(move, superinfo_t(0))).second) moves.push_back(move);
  }

  void child(side_t move, superinfo_t info) {
    if (children.insert(make_pair(move, info)).second) moves.push_back(move);
  }

  results_t return_(superinfo_t info) const { return *this; }
};

// Evaluate a position for all important rotations, returning the set of rotations in which we win.
// Note that a "win" for white includes ties.  The returned set of known rotations definitely includes
// all important rotations, but may include others as well.
template<bool remember,bool aggressive,bool debug>
__attribute__((noinline)) static typename results_t<remember>::type
super_evaluate_recurse(const int depth, const side_t side0, const side_t side1, superdata_t data,
                       const super_t important) {
  STAT(total_expanded_nodes++);
  STAT(expanded_nodes[depth]++);
  PRINT_STATS(24);
  superinfo_t& info = data.lookup.info;
  super_t possible = 0; // Keep track of possible wins that we might have missed
  results_t<remember> results; // Optionally keep track of results about children

  // Consistency check
  GEODE_ASSERT(info.valid());

  // Be verbose if desired
  TRACE_VERBOSE_START(depth, pack(side0, side1));
  TRACE_VERBOSE("input %s %s", subset(info.known, verbose), subset(info.wins, verbose));

  // Collect possible moves
  SIMPLE_MOVES(side0, side1)
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
    if (aggressive) {
      super_t immediate = ours&~theirs;
      info.wins |= immediate&~info.known;
      info.known |= immediate;
      results.immediate_win(move,immediate);
    }

    // Do a shallow evaluation of the child position
    const super_t mask = rmax(important&~info.known);
    children[i] = super_shallow_evaluate<!aggressive,debug>(depth-1,side1,move,theirs,ours,mask);
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
    int order[36]; // We'll sort moves in ascending order of this array.  Can't be order[total] due to compiler bug.
    for (int i=total-1;i>=0;i--) {
      int closeness = aggressive?arbitrarily_rotated_win_closeness(moves[i],side1)
                                :arbitrarily_rotated_win_closeness(side1,moves[i]);
      if (!remember) {
        int distance = 6-(closeness>>16);
        if (distance==6 || distance>(depth-aggressive)/2) { // We can't reach a black win within the given search depth
          if (!aggressive) {
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
      order[i] = aggressive?-closeness:closeness;
    }

    // Are we out of recursion depth?
    if (depth==1 && total) {
      TRACE_VERBOSE("out of recursion depth");
      goto done;
    }

    // Sort moves and children arrays based on order
    insertion_sort(total,order,moves,children);

    // Optionally limit black's choices
    if (aggressive)
      total = min(total,super_move_limit);

    // Recurse
    for (int i=0;i<total;i++) {
      const side_t move = moves[i];
      super_t mask = rmax(important&~info.known);
      superinfo_t child = super_evaluate_recurse<false,!aggressive,debug>(depth-1,side1,move,children[i],mask);
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
    super_store<aggressive>(depth,data.lookup);
  return results.return_(info);
}

template superinfo_t super_evaluate_recurse<false>(bool,int,side_t,side_t,superdata_t,super_t);

// Use only from top level
template<bool remember> typename results_t<remember>::type
super_evaluate_recurse(const bool aggressive, const int depth, const side_t side0, const side_t side1,
                       superdata_t data, const super_t important) {
  #define CASE(a,d) \
    if (aggressive==a && debug==d) \
      return super_evaluate_recurse<remember,a,d>(depth, side0, side1, data, important);
  CASE(0,0) CASE(0,1) CASE(1,0) CASE(1,1)
  __builtin_unreachable();
}

static inline score_t to_score(bool aggressive, int depth, bool win) {
  int value = aggressive+win;
  return value==1?score(depth,value):exact_score(value);
}

// Driver for evaluation abstracted over rotations
score_t super_evaluate(bool aggressive, int depth, const board_t board, const Vector<int,4> rotation) {
  // We can afford error detection here since recursion happens into a different function
  GEODE_ASSERT(supertable_bits()>=10);
  GEODE_ASSERT(depth>=0);
  check_board(board);

  // Unpack board
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);

  // We don't need to search past the end of the game
  depth = min(depth,36-popcount(side0|side1));

  // Exit immediately if possible
  const super_t start = super_t::singleton(rotation);
  const superdata_t data = super_shallow_evaluate(aggressive,depth,side0,side1,start);
  if (data.lookup.info.known(rotation))
    return to_score(aggressive,depth,data.lookup.info.wins(rotation));
  if (!depth)
    return score(0,1);

  // Otherwise, recurse into children
  const superinfo_t info = super_evaluate_recurse<false>(aggressive,depth,side0,side1,data,start);
  GEODE_ASSERT(info.known(rotation));
  return to_score(aggressive,depth,info.wins(rotation));
}

super_t super_evaluate_all(bool aggressive, int depth, const board_t board) {
  // We can afford error detection here since recursion happens into a different function
  GEODE_ASSERT(supertable_bits()>=10);
  GEODE_ASSERT(depth>=0);
  check_board(board);

  // Unpack board
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);

  // We don't need to search past the end of the game
  depth = min(depth,36-popcount(side0|side1));

  // Exit immediately if possible
  const superdata_t data = super_shallow_evaluate(aggressive,depth,side0,side1,~super_t(0));
  if (!~data.lookup.info.known)
    return data.lookup.info.wins;
  if (!depth)
    return aggressive?data.lookup.info.wins:data.lookup.info.wins|~data.lookup.info.known;

  // Otherwise, recurse into children
  const superinfo_t info = super_evaluate_recurse<false>(aggressive,depth,side0,side1,data,~super_t(0));
  GEODE_ASSERT(!~info.known);
  return info.wins;
}

// Evaluate enough children to determine who wins, and return the results
vector<tuple<rotated_board_t,score_t>>
super_evaluate_children(const bool aggressive, const int depth, const board_t board, const Vector<int,4> rotation) {
  // We can afford error detection here since recursion happens into a different function
  GEODE_ASSERT(supertable_bits()>=10);
  GEODE_ASSERT(depth>=1);
  check_board(board);

  // Unpack board
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);

  // We don't need to search past the end of the game
  const int infinite_depth = 36-popcount(side0|side1),
            search_depth = min(depth,infinite_depth),
            report_depth = depth>=infinite_depth?36:depth;

  // If we already know the value of this position, super_evaluate_recurse won't give us any child information.
  // Therefore, pretend we don't know.
  const super_t start = super_t::singleton(rotation);
  superdata_t data = super_shallow_evaluate(aggressive,search_depth,side0,side1,start);
  data.lookup.info.wins &= ~start;
  data.lookup.info.known &= ~start;

  // Recurse into children
  const results_t<true> results = super_evaluate_recurse<true>(aggressive,search_depth,side0,side1,data,start);
  GEODE_ASSERT(results.moves.size());

  // Analyze results
  vector<tuple<rotated_board_t,score_t>> children;
  for (const side_t move : results.moves) {
    const board_t after = pack(side1,move);
    // If there's an immediate win, generate a move with no rotation.
    // Such moves are legal only in this case.
    const auto it = results.immediate_wins.find(move);
    if (it != results.immediate_wins.end() && it->second(rotation))
      children.push_back(make_tuple(make_tuple(after,rotation),to_score(aggressive,report_depth,true)));
    else {
      // Otherwise, report results for all known children
      const superinfo_t& info = check_get(results.children, move);
      for (Vector<int,4> r : single_rotations)
        if (info.known(rotation+r)) {
          children.push_back(make_tuple(make_tuple(after,rotation+r),to_score(aggressive,report_depth,!info.wins(rotation+r))));
        }
    }
  }
  GEODE_ASSERT(children.size());
  return children;
}

}
