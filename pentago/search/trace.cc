// Tracing code for use in debugging inconsistency detections

#include "pentago/search/trace.h"
#include "pentago/search/superengine.h"
#include "pentago/base/moves.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/log.h"
#include <unordered_set>
#include <unordered_map>
namespace pentago {

using std::tie;
using std::get;
using std::make_tuple;
using std::unordered_set;
using std::unordered_map;

struct situation_t {
  bool aggressive;
  int depth;
  board_t board;

  situation_t(bool aggressive, int depth, board_t board)
    : aggressive(aggressive)
    , depth(depth)
    , board(board) {
    check_board(board);
  }
};

// Persistent trace information
static unordered_set<board_t> board_flags; // Map from traced boards to depth
static unordered_map<tuple<int,board_t>,superinfo_t,
                     boost::hash<tuple<int,board_t>>> known; // Known information at various depths
static vector<situation_t> errors; // Potentially problematic aggressive,depth,boards triples
static vector<situation_t> dependencies; // Aggressive,depth,board triples that we depend on
static int stone_depth; // Current search depth + stone count

void trace_restart() {
  stone_depth = -1;
  errors.clear();
}

bool traced(bool aggressive, board_t board) {
  return contains(board_flags, get<0>(superstandardize(board))|(uint64_t)aggressive<<aggressive_bit);
}

static int trace_verbose_depth = 0;

string trace_verbose_prefix() {
  return format("%*strace verbose: ",trace_verbose_depth,"");
}

Array<uint8_t> trace_verbose_start(int depth, board_t board) {
  symmetry_t symmetry;
  tie(board, symmetry) = superstandardize(board);
  vector<uint8_t> watch;
  if (depth==5 && board==35466671711863625)
    watch.push_back(89);
  else if (depth==4 && board==39688947336609933) {
    watch.push_back(113);
    watch.push_back(49);
  } else
    return asarray(watch).copy();
  // Find r and global g s.t. r(b) = gws(b)
  vector<int> iwatch;
  for (auto c : watch)
    iwatch.push_back(c);
  slog("%sdepth %d, board %d, rotations %s, stones %d",
       trace_verbose_prefix(), depth, board, asarray(iwatch), count_stones(board));
  trace_verbose_depth += 2;
  for (uint8_t& w : watch) {
    symmetry_t s = symmetry_t(symmetry.global,0).inverse()*symmetry_t(0,w)*symmetry;
    GEODE_ASSERT(!s.global);
    w = s.local;
  }
  return asarray(watch).copy();
}

void trace_verbose_end() {
  trace_verbose_depth -= 2;
}

string subset(super_t s, RawArray<const uint8_t> w) {
  string result;
  for (auto r : w)
    result += "01"[s(r)];
  return result;
}

void trace_error(bool aggressive, int depth, board_t board, const char* context) {
  check_board(board);
  board = get<0>(superstandardize(board));
  errors.push_back(situation_t(aggressive,depth,board));
  THROW(RuntimeError,"trace inconsistency detected in %s: depth %d, board %lld, aggressive %d, stone depth %d",context,depth,board,aggressive,depth+count_stones(board));
}

void trace_dependency(bool parent_aggressive, int parent_depth, board_t parent, bool child_aggressive,
                      int child_depth, board_t child, superinfo_t child_info) {
  check_board(parent);
  check_board(child);
  if (traced(parent_aggressive,parent))
    dependencies.push_back(situation_t(child_aggressive, child_depth, get<0>(superstandardize(child))));
}

void trace_check(bool aggressive, int depth, board_t board, superinfo_t info, const char* context) {
  check_board(board);
  // Stone depth should increase monotonically
  GEODE_ASSERT(stone_depth<=depth+count_stones(board));
  stone_depth = depth+count_stones(board);
  // Standardize board
  symmetry_t symmetry;
  tie(board, symmetry) = superstandardize(board);
  info.known = transform_super(symmetry,info.known);
  info.wins = transform_super(symmetry,info.wins);
  // Compare with what we know
  const superinfo_t* known_info = get_pointer(known, make_tuple(depth,board));
  if (known_info) {
    const super_t errors = (known_info->wins^info.wins)&known_info->known&info.known;
    if (errors) {
      const uint8_t r = first(errors);
      slog("trace check failed in %s: depth %d, board %lld, aggressive %d, stones %d, rotation %d, "
           "correct %d, got %d", context, depth, board, aggressive, count_stones(board), r,
           known_info->wins(r), info.wins(r));
      trace_error(aggressive,depth,board,format("%s.trace_check",context).c_str());
    }
  }
}

static void clean_evaluate(bool aggressive, int depth, board_t board) {
  check_board(board);
  symmetry_t symmetry;
  tie(board, symmetry) = superstandardize(board);
  if (contains(known, make_tuple(depth,board)))
    return;
  slog("clean evaluate: depth %d, board %d", depth, board);
  clear_supertable();
  const super_t all = ~super_t(0);
  const side_t side0 = unpack(board,0), side1 = unpack(board,1);
  auto data = super_shallow_evaluate(aggressive,depth,side0,side1,all);
  superinfo_t info = data.lookup.info;
  if (depth>=1)
    info = super_evaluate_recurse<false>(aggressive,depth,side0,side1,data,all);
  GEODE_ASSERT(info.known==all);
  known[make_tuple(depth,board)] = info;
}

static super_t known_wins(int depth, board_t board) {
  check_board(board);
  symmetry_t symmetry;
  tie(board, symmetry) = superstandardize(board);
  superinfo_t info = check_get(known, make_tuple(depth,board));
  GEODE_ASSERT(!~info.known);
  return transform_super(symmetry.inverse(),info.wins);
}

static string str_move(side_t move) {
  int n = integer_log_exact(move);
  if (move!=((side_t)1<<n))
    return format("bad(%d)",move);
  int q = n/16, i = n%16;
  int x = 3*(q/2)+i/3,
      y = 3*(q%2)+i%3;
  return format("%c%c","fedcba"[y],"123456"[x]);
}

// Do a clean evaluation of each board involved in an error or dependency
bool trace_learn() {
  // Copy errors and dependencies since they'll be cleared during clean evaluation
  auto errors = pentago::errors;
  auto dependencies = pentago::dependencies;

  // Learn about each error and each dependency
  const int count = known.size();
  for (auto error : errors)
    clean_evaluate(error.aggressive,error.depth,error.board);
  for (auto dep : dependencies)
    clean_evaluate(dep.aggressive,dep.depth,dep.board);

  // If we haven't learned anything new, evaluate the children of each error
  if (size_t(count) < known.size())
    return true;
  for (auto error : errors) {
    const bool aggressive = error.aggressive;
    const int depth = error.depth;
    GEODE_ASSERT(depth);
    const board_t board = error.board;
    const side_t side0 = unpack(board,0), side1 = unpack(board,1);

    // Evaluate all dependencies
    SIMPLE_MOVES(side0,side1);
    for (int i=0;i<total;i++)
      clean_evaluate(!aggressive,depth-1,pack(side1,moves[i]));

    // Verify that we're consistent with the child values
    super_t wins = 0;
    for (int r=0;r<256;r++) {
      const symmetry_t s(0,r);
      const bool verbose =    (depth==5 && board==35466671711863625 && r==89)
                           || (depth==4 && board==39688947336609933 && r==113);
      board_t rb = transform_board(s,board);
      board_t rside0 = transform_side(s,side0);
      if (verbose)
        slog("\n\n\ntrace_learn: checking board %d:\n%s", rb, str_board(rb));
      int st = status(rb);
      if (verbose)
        slog("status = %d", st);
      if (st)
        wins |= (aggressive?st==1:st!=2) ? super_t::singleton(r) : super_t(0);
      else
        for (int i=0;i<total;i++) {
          const side_t rmove = transform_side(s,moves[i]);
          if (won(rmove)) {
            if (verbose)
              slog("immediate win with move %s", str_move(rmove^rside0));
            wins |= super_t::singleton(r);
            goto win;
          } else {
            board_t child = pack(side1,moves[i]);
            super_t child_wins = known_wins(depth-1,child);
            for (int q=0;q<4;q++) for (int d=1;d<=3;d+=2) {
              symmetry_t cs = symmetry_t(0,d<<2*q)*s;
              if (!child_wins(cs.local)) {
                if (verbose)
                  slog("win with move %s, rotation %d %d", str_move(rmove^rside0), q, d);
                wins |= super_t::singleton(r);
                goto win;
              }
            }
          }
        }
      win:
      if (verbose) {
        if (!wins(r))
          slog("loss");
        slog("\n\n");
      }
    }
    super_t parent_wins = known_wins(depth,board);
    super_t errors = wins^parent_wins;
    if (errors) {
      uint8_t r = first(errors);
      slog("trace_learn: inconsistent results for clean evaluation of depth %d, board %lld, "
           "aggressive %d: rotation %d, parent %d, children %d", depth, board, aggressive, r,
           parent_wins(r), wins(r));
      return false;
    }
  }
 
  // If we still haven't learned anything new, any errors are immediate
  if (size_t(count) < known.size())
    return true;
  for (auto error : errors)
    slog("trace_learn: isolated error for depth %d, board %lld, aggressive %d", error.depth,
         error.board, error.aggressive);
  return false;
}

}
