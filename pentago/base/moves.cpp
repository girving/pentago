// Move generation code

#include <pentago/base/moves.h>
#include <geode/array/Array.h>
#include <geode/python/wrap.h>
namespace pentago {

using namespace geode;

static Array<board_t> moves(board_t board) {
  check_board(board);
  // const board_t moves[total] = {...};
  MOVES(board)
  return RawArray<board_t>(total,moves).copy();
}

static Array<board_t> simple_moves(board_t board) {
  check_board(board);
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  // const side_t moves[total] = {...};
  SIMPLE_MOVES(side0,side1)
  Array<board_t> result(total,false);
  for (int i=0;i<total;i++)
    result[i] = pack(side1,moves[i]);
  return result;
}

}
using namespace pentago;
using namespace geode::python;

void wrap_moves() {
  GEODE_FUNCTION(moves)
  GEODE_FUNCTION(simple_moves)
}
