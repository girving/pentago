// Move generation code

#include "pentago/base/moves.h"
#include "pentago/utility/array.h"
namespace pentago {

Array<board_t> moves(board_t board) {
  check_board(board);
  // const board_t moves[total] = {...};
  MOVES(board)
  return RawArray<board_t>(total, moves).copy();
}

Array<board_t> simple_moves(board_t board) {
  check_board(board);
  const side_t side0 = unpack(board,0),
               side1 = unpack(board,1);
  // const side_t moves[total] = {...};
  SIMPLE_MOVES(side0,side1)
  Array<board_t> result(total,uninit);
  for (int i=0;i<total;i++)
    result[i] = pack(side1,moves[i]);
  return result;
}

}
