// High level interface to board

#include "pentago/high/board.h"
#include "pentago/base/score.h"
#include "pentago/base/symmetry.h"
#include "pentago/search/superengine.h"
#include "pentago/utility/random.h"
#include "pentago/utility/const_cast.h"
namespace pentago {

using std::max;

high_board_t::high_board_t(const board_t board, const bool middle)
  : board(board)
  , turn(0) // Filled in below
  , middle(middle)
  , grid(to_table(board)) {
  // Check correctness
  check_board(board);
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const int count0 = popcount(side0),
            count1 = popcount(side1);
  GEODE_ASSERT(count0+count1==count_stones(board));
  const_cast_(turn) = count0-1==count1-middle;
  if (count0-turn-middle*(turn==0)!=count1-middle*(turn==1) || (middle && !board))
    throw ValueError(format("high_board_t: inconsistent board %lld, turn %d, middle %d, "
                            "side counts %d %d", board, turn, middle, count0, count1));
}

high_board_t::~high_board_t() {}

int high_board_t::count() const {
  return count_stones(board);
}

bool high_board_t::done() const {
  return won(unpack(board,0))
      || won(unpack(board,1))
      || (!middle && count_stones(board)==36);
}

vector<high_board_t> high_board_t::moves() const {
  vector<high_board_t> moves;
  if (!middle) { // Place a stone
    for (const int x : range(6))
      for (const int y : range(6))
        if (!grid(x,y))
          moves.push_back(place(x,y));
  } else { // Rotate a quadrant
    for (const int qx : range(2))
      for (const int qy : range(2))
        for (const int d : vec(-1,1))
          moves.push_back(rotate(qx,qy,d));
  }
  return moves;
}

high_board_t high_board_t::place(const int x, const int y) const {
  GEODE_ASSERT(!middle);
  GEODE_ASSERT(0<=x && x<6);
  GEODE_ASSERT(0<=y && y<6);
  GEODE_ASSERT(!grid(x,y));
  return high_board_t(
    board+flip_board(pack(side_t(1<<(3*(x%3)+y%3))<<16*(2*(x/3)+(y/3)),side_t(0)),turn),
    true);
}

high_board_t high_board_t::rotate(const int qx, const int qy, const int d) const {
  GEODE_ASSERT(middle);
  GEODE_ASSERT(qx==0 || qx==1);
  GEODE_ASSERT(qy==0 || qy==1);
  GEODE_ASSERT(abs(d)==1);
  return high_board_t(
    transform_board(symmetry_t(local_symmetry_t((d>0?1:3)<<2*(2*qx+qy))),board),
    false);
}

// If aggressive is true, true is win, otherwise true is win or tie.
static bool exact_lookup(const bool aggressive, const board_t board, const block_cache_t& cache) {
  {
    super_t wins;
    if (cache.lookup(aggressive,board,wins))
      return wins(0);
  }
  // Fall back to tree search
  const score_t score = super_evaluate(aggressive,100,board,Vector<int,4>());
  return (score&3)>=aggressive+1;
}

int high_board_t::immediate_value() const {
  GEODE_ASSERT(done());
  const bool bw = won(unpack(board,0)),
             ww = won(unpack(board,1));
  if (bw || ww)
    return bw && ww ? 0 : bw==!turn ? 1 : -1;
  return 0;
}

int high_board_t::value(const block_cache_t& cache) const {
  if (done())
    return immediate_value();
  else if (!middle) {
    // If we're not at a half move, look up the result
    const auto flip = flip_board(board,turn);
    return exact_lookup(true,flip,cache)+exact_lookup(false,flip,cache)-1;
  } else {
    int best = -1;
    for (const auto& move : moves()) {
      best = max(best, -move.value(cache));
      if (best==1)
        break;
    }
    return best;
  }
}

int high_board_t::value_check(const block_cache_t& cache) const {
  GEODE_ASSERT(!middle);
  if (done())
    return immediate_value();
  {
    int value = -1;
    for (const auto& move : moves())
      value = max(value, move.value(cache));
    const int lookup = this->value(cache);
    if (value != lookup)
      throw AssertionError(format(
          "high_board_t.value_check: board %lld, turn %d, middle %d, computed %d != lookup %d\n%s",
          board, turn, middle, value, lookup, str_board(board)));
    return value;
  }
}

Vector<int,3> high_board_t::sample_check(const block_cache_t& cache, RawArray<const board_t> boards,
                                         RawArray<const Vector<super_t,2>> wins) {
  GEODE_ASSERT(boards.size()==wins.size());
  Vector<int,3> counts;
  Random random(152311341);
  for (const int i : range(boards.size())) {
    const int r = random.bits<uint8_t>();
    const int n = count_stones(boards[i]);
    const high_board_t board(transform_board(symmetry_t(local_symmetry_t(r)), boards[i]), false);
    const int value = board.value_check(cache);
    GEODE_ASSERT(value == wins[i][n&1](r)-wins[i][!(n&1)](r));
    counts[value+1]++;
  }
  return counts;
}

string high_board_t::name() const {
  return format("%d%s", board, middle ? "m" : "");
}

ostream& operator<<(ostream& output, const high_board_t& board) {
  output << board.board;
  if (board.middle)
    output << 'm';
  return output;
}

high_board_t high_board_t::parse(const string& name) {
  static_assert(sizeof(long long) == 8);
  if (name.size() && isdigit(name[0]) && (isdigit(name.back()) || name.back()=='m')) {
    char* end;
    const board_t board = strtoll(name.c_str(), &end, 0);
    const auto left = name.c_str()+name.size()-end;
    if (left==0 || (left==1 && name.back()=='m'))
      return high_board_t(board, left==1);
  }
  throw ValueError(format("high_board_t: invalid board name '%s'", name));
}

}
