// Endgame database computation
#pragma once

#include <pentago/base/board.h>
#include <pentago/base/superscore.h>
namespace pentago {

void endgame_verify_board(const char* prefix, const board_t board, const Vector<super_t,2>& result, bool verbose=false);

}
