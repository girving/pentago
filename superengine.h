// Core tree search engine abstracted over rotations

#include "supertable.h"
#include <vector>
namespace pentago {

// For historical reasons, we call the player who wins ties white, and the other player black,
// even the code is also capable of computing in the other direction.

using std::vector;

// Limit black's options to the given number of moves (chosen after move ordering)
void set_super_move_limit(int limit);

// Turn on or off debug mode
void set_super_debug(bool on);

// Current knowledge about a board position.  This is computed during shallow evaluation and
// passed down to super_evaluate_recurse if the node is expanded.  It also sucks up rather a
// lot of stack space, which may turn into an issue if we ever port to a weird platform.
struct superdata_t {
  superlookup_t lookup;
  super_t wins1; // super_wins(side1)
};

// Evaluate everything we can about a position without recursing into children.
superdata_t super_shallow_evaluate(const bool aggressive, const int depth, const side_t side0, const side_t side1, const super_t interesting);

// For most of the tree we care only about the value of a node, but at the top we may want at least one optimal move as well.  In order to
// avoid two separate routines with a bunch of duplicate logic, we use the following templatized helper.
template<bool remember> struct results_t;
template<> struct results_t<false> {
  typedef superinfo_t type;
  static void immediate_win(side_t move, super_t immediate) {}
  static void child(side_t move, superinfo_t info) {}
  static superinfo_t return_(superinfo_t info) { return info; }
};

// Evaluate a position for all important rotations, returning the set of rotations in which we win.
// Note that a "win" for white includes ties.  The returned set of known rotations definitely includes
// all important rotations, but may include others as well.
template<bool remember> typename results_t<remember>::type super_evaluate_recurse(const bool aggressive, const int depth, const side_t side0, const side_t side1, superdata_t data, const super_t important);

// Driver for evaluation abstracted over rotations
score_t super_evaluate(bool aggressive, int depth, const board_t board, const Vector<int,4> rotation);

// Evaluate the result of all possible rotations of a position
super_t super_evaluate_all(bool aggressive, int depth, const board_t board);

typedef Tuple<board_t,Tuple<int,int,int,int>> rotated_board_t;

// Evaluate enough children to determine who wins, and return the results
vector<Tuple<rotated_board_t,score_t>> super_evaluate_children(const bool aggressive, const int depth, const board_t board, const Vector<int,4> rotation);

}
