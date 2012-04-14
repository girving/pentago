// Tracing code for use in debugging inconsistency detections

#include "trace.h"
#include "moves.h"
#include "superengine.h"
#include <other/core/python/module.h>
#include <other/core/structure/Hashtable.h>
namespace pentago {

using std::cout;
using std::endl;

// Persistent trace information
static Hashtable<board_t> boards; // Map from traced boards to depth
static Hashtable<Tuple<int,board_t>,superinfo_t> known; // Known information at various depths
static Array<Tuple<int,board_t>> errors; // Potentially problematic depth,boards pairs
static Array<Tuple<int,board_t>> dependencies; // 
static int stone_depth; // Current search depth + stone count

void trace_restart() {
  stone_depth = -1;
  errors.remove_all();
}

bool traced(board_t board) {
  return boards.contains(superstandardize(board).x);
}

static int trace_verbose_depth = 0;

Array<uint8_t> trace_verbose_start(int depth, board_t board) {
  symmetry_t symmetry;
  superstandardize(board).get(board,symmetry);
  Array<uint8_t> watch;
  if (depth==5 && board==35466671711863625)
    watch.append(89);
  else if (depth==4 && board==39688947336609933) {
    watch.append(113);
    watch.append(49);
  } else
    return watch;
  // Find r and global g s.t. r(b) = gws(b)
  Array<int> iwatch;
  for (auto c : watch)
    iwatch.append(c);
  cout << trace_verbose_prefix()<<"depth "<<depth<<", board "<<board<<", rotations "<<iwatch<<", stones "<<count_stones(board)<<endl;
  trace_verbose_depth += 2;
  for (uint8_t& w : watch) {
    symmetry_t s = symmetry_t(symmetry.global,0).inverse()*symmetry_t(0,w)*symmetry;
    OTHER_ASSERT(!s.global);
    w = s.local;
  }
  return watch;
}

string trace_verbose_prefix() {
  return format("%*strace verbose: ",trace_verbose_depth,"");
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

void trace_error(int depth, board_t board, const char* context) {
  board = superstandardize(board).x;
  errors.append(tuple(depth,board));
  throw RuntimeError(format("trace inconsistency detected in %s: depth %d, board %lld, black %d, stone depth %d",context,depth,board,black_to_move(board),depth+count_stones(board)));
}

void trace_dependency(int parent_depth, board_t parent, int child_depth, board_t child, superinfo_t child_info) {
  if (traced(parent))
    dependencies.append(tuple(child_depth,superstandardize(child).x));
}

void trace_check(int depth, board_t board, superinfo_t info, const char* context) {
  // Stone depth should increase monotonically
  OTHER_ASSERT(stone_depth<=depth+count_stones(board));
  stone_depth = depth+count_stones(board);
  // Standardize board
  symmetry_t symmetry;
  superstandardize(board).get(board,symmetry);
  info.known = transform_super(symmetry,info.known);
  info.wins = transform_super(symmetry,info.wins);
  // Compare with what we know
  const superinfo_t* known_info = known.get_pointer(tuple(depth,board));
  const super_t errors = known_info?(known_info->wins^info.wins)&known_info->known&info.known:super_t(0);
  if (errors) {
    const uint8_t r = first(errors);
    cout << format("trace check failed in %s: depth %d, board %lld, black %d, stones %d, rotation %d, correct %d, got %d",
      context,depth,board,black_to_move(board),count_stones(board),r,known_info->wins(r),info.wins(r))<<endl;
    trace_error(depth,board,format("%s.trace_check",context).c_str());
  }
}

static void clean_evaluate(int depth, board_t board) {
  symmetry_t symmetry;
  superstandardize(board).get(board,symmetry);
  if (known.contains(tuple(depth,board)))
    return;
  cout << "clean evaluate: depth "<<depth<<", board "<<board<<endl;
  clear_supertable();
  const super_t all = ~super_t(0);
  const side_t side0 = unpack(board,0), side1 = unpack(board,1);
  auto data = super_shallow_evaluate(depth,side0,side1,all);
  superinfo_t info = data.lookup.info;
  if (depth>=1)
    info = super_evaluate_recurse<false>(depth,side0,side1,data,all);
  OTHER_ASSERT(info.known==all);
  known.set(tuple(depth,board),info);
}

static super_t known_wins(int depth, board_t board) {
  symmetry_t symmetry;
  superstandardize(board).get(board,symmetry);
  superinfo_t info = known.get(tuple(depth,board));
  OTHER_ASSERT(!~info.known);
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
static bool trace_learn() {
  // Copy errors and dependencies since they'll be cleared during clean evaluation
  auto errors = pentago::errors;
  auto dependencies = pentago::dependencies;

  // Learn about each error and each dependency
  int count = known.size();
  for (auto error : errors)
    clean_evaluate(error.x,error.y);
  for (auto dep : dependencies)
    clean_evaluate(dep.x,dep.y);

  // If we haven't learned anything new, evaluate the children of each error
  if (count<known.size())
    return true;
  for (auto error : errors) {
    const int depth = error.x;
    OTHER_ASSERT(depth);
    const board_t board = error.y;
    const side_t side0 = unpack(board,0), side1 = unpack(board,1);
    const bool black = black_to_move(board);

    // Evaluate all dependencies
    SIMPLE_MOVES(side0,side1);
    for (int i=0;i<total;i++)
      clean_evaluate(depth-1,pack(side1,moves[i]));

    // Verify that we're consistent with the child values
    super_t wins = 0;
    for (int r=0;r<256;r++) {
      const symmetry_t s(0,r);
      const bool verbose =    (depth==5 && board==35466671711863625 && r==89)
                           || (depth==4 && board==39688947336609933 && r==113);
      board_t rb = transform_board(s,board);
      board_t rside0 = transform_side(s,side0);
      if (verbose)
        cout << "\n\n\ntrace_learn: checking board "<<rb<<":\n"<<str_board(rb)<<endl;
      int st = status(rb);
      if (verbose)
        cout << "status = "<<st<<endl;
      if (st)
        wins |= (black?st==1:st!=2) ? super_t::singleton(r) : super_t(0);
      else
        for (int i=0;i<total;i++) {
          const side_t rmove = transform_side(s,moves[i]);
          if (won(rmove)) {
            if (verbose)
              cout << "immediate win with move "<<str_move(rmove^rside0)<<endl;
            wins |= super_t::singleton(r);
            goto win;
          } else {
            board_t child = pack(side1,moves[i]);
            super_t child_wins = known_wins(depth-1,child);
            for (int q=0;q<4;q++) for (int d=1;d<=3;d+=2) {
              symmetry_t cs = symmetry_t(0,d<<2*q)*s;
              if (!child_wins(cs.local)) {
                if (verbose) {
                  cout << "win with move "<<str_move(rmove^rside0)<<", rotation "<<q<<' '<<d<<endl;
                  //cout << "rside0 =\n"<<str_board(pack(rside0,(side_t)0))<<endl;
                  //cout << "rmove =\n"<<str_board(pack(rmove,(side_t)0))<<endl;
                }
                wins |= super_t::singleton(r);
                goto win;
              }
            }
          }
        }
      win:
      if (verbose) {
        if (!wins(r))
          cout << "loss"<<endl;
        cout << "\n\n\n";
      }
    }
    super_t parent_wins = known_wins(depth,board);
    super_t errors = wins^parent_wins;
    if (errors) {
      uint8_t r = first(errors);
      cout << format("trace_learn: inconsistent results for clean evaluation of depth %d, board %lld, black %d: rotation %d, parent %d, children %d",depth,board,black_to_move(board),r,parent_wins(r),wins(r))<<endl;
      return false;
    }
  }
 
  // If we still haven't learned anything new, any errors are immediate
  if (count<known.size())
    return true;
  for (auto error : errors)
    cout << format("trace_learn: isolated error for depth %d, board %lld, black %d",error.x,error.y,black_to_move(error.y))<<endl;
  return false;
}

}
using namespace pentago;

void wrap_trace() {
  OTHER_FUNCTION(trace_learn)
}
