// High level interface to board

#include <pentago/high/board.h>
#include <pentago/base/score.h>
#include <pentago/base/symmetry.h>
#include <geode/python/Class.h>
#include <geode/python/stl.h>
#include <geode/random/Random.h>
namespace pentago {

GEODE_DEFINE_TYPE(high_board_t)

high_board_t::high_board_t(const board_t board, const int turn, const bool middle)
  : board(board)
  , turn(turn)
  , middle(middle)
  , grid(to_table(board)) {
  // Check correctness
  check_board(board);
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  const int count0 = popcount(side0),
            count1 = popcount(side1);
  GEODE_ASSERT(count0+count1==count_stones(board));
  GEODE_ASSERT(turn==0 || turn==1);
  if (count0-turn-middle*(turn==0)!=count1-middle*(turn==1))
    throw ValueError(format("high_board_t: inconsistent board %lld, turn %d, middle %d, side counts %d %d",
      board,turn,middle,count0,count1));
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

vector<Ref<high_board_t>> high_board_t::moves() const {
  GEODE_ASSERT(!done());
  vector<Ref<high_board_t>> moves;
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

Ref<high_board_t> high_board_t::place(const int x, const int y) const {
  GEODE_ASSERT(!done());
  GEODE_ASSERT(!middle);
  GEODE_ASSERT(0<=x && x<6);
  GEODE_ASSERT(0<=y && y<6);
  GEODE_ASSERT(!grid(x,y));
  return new_<high_board_t>(
    board+flip_board(pack(side_t(1<<(3*(x%3)+y%3))<<16*(2*(x/3)+(y/3)),side_t(0)),turn),
    turn,true);
}

Ref<high_board_t> high_board_t::rotate(const int qx, const int qy, const int d) const {
  GEODE_ASSERT(!done());
  GEODE_ASSERT(middle);
  GEODE_ASSERT(qx==0 || qx==1);
  GEODE_ASSERT(qy==0 || qy==1);
  GEODE_ASSERT(abs(d)==1);
  return new_<high_board_t>(
    transform_board(symmetry_t(local_symmetry_t((d>0?1:3)<<2*(2*qx+qy))),board),
    !turn,false);
}

int high_board_t::value(const block_cache_t& cache) const {
  {
    // Check immediate wins
    const bool bw = won(unpack(board,0)),
               ww = won(unpack(board,1));
    if (bw || ww)
      return bw && ww ? 0 : bw==!turn ? 1 : -1;
    if (!middle && count_stones(board)==36)
      return 0;
  }
  if (!middle) {
    // If we're not at a half move, look up the result
    const auto flip = flip_board(board,turn);
    super_t we_win, we_win_or_tie;
    GEODE_ASSERT(cache.lookup(true,flip,we_win));
    GEODE_ASSERT(cache.lookup(false,flip,we_win_or_tie));
    return we_win(0)+we_win_or_tie(0)-1;
  } else {
    int best = -1;
    for (const auto move : moves())
      best = max(best,-move->value(cache));
    return best;
  }
}

int high_board_t::value_check(const block_cache_t& cache) const {
  GEODE_ASSERT(!middle);
  {
    // Check immediate wins
    const bool bw = won(unpack(board,0)),
               ww = won(unpack(board,1));
    if (bw || ww)
      return bw && ww ? 0 : bw==!turn ? 1 : -1;
    if (!middle && count_stones(board)==36)
      return 0;
  } {
    int value = -1;
    for (const auto& move : moves())
      value = max(value,move->value(cache));
    GEODE_ASSERT(value==this->value(cache));
    return value;
  }
}

void high_board_t::sample_check(const block_cache_t& cache, RawArray<const board_t> boards, RawArray<const Vector<super_t,2>> wins) {
  GEODE_ASSERT(boards.size()==wins.size());
  const auto random = new_<Random>(152311341);
  for (const int i : range(boards.size())) {
    const int r = random->bits<uint8_t>();
    const int n = count_stones(boards[i]);
    const auto board = new_<high_board_t>(transform_board(symmetry_t(local_symmetry_t(r)),boards[i]),n&1,false);
    GEODE_ASSERT(board->value_check(cache)==wins[i][n&1](r)-wins[i][!(n&1)](r));
  }
}

}
using namespace pentago;

void wrap_high_board() {
  typedef high_board_t Self;
  Class<Self>("high_board_t")
    .GEODE_INIT(board_t,int,bool)
    .GEODE_FIELD(board)
    .GEODE_FIELD(turn)
    .GEODE_FIELD(middle)
    .GEODE_FIELD(grid)
    .GEODE_GET(count)
    .GEODE_METHOD(done)
    .GEODE_METHOD(moves)
    .GEODE_METHOD(place)
    .GEODE_METHOD(rotate)
    .GEODE_METHOD(value)
    .GEODE_METHOD(value_check)
    .GEODE_METHOD(sample_check)
    .compare()
    ;
}
