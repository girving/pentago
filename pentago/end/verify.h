// Endgame database verification

#include "pentago/base/board.h"
#include "pentago/base/superscore.h"
#include "pentago/utility/array.h"
#include "pentago/utility/random.h"
namespace pentago {

void endgame_sparse_verify(RawArray<const board_t> boards, RawArray<const Vector<super_t,2>> wins,
                           Random& random, int samples);

}
