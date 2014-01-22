// Endgame database verification

#include <pentago/base/count.h>
#include <pentago/base/section.h>
#include <pentago/base/superscore.h>
#include <pentago/base/symmetry.h>
#include <pentago/data/supertensor.h>
#include <pentago/search/superengine.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/convert.h>
#include <pentago/utility/debug.h>
#include <geode/array/Array2d.h>
#include <geode/array/Array4d.h>
#include <geode/math/integer_log.h>
#include <geode/python/wrap.h>
#include <geode/python/Class.h>
#include <geode/python/ExceptionValue.h>
#include <geode/random/Random.h>
#include <geode/utility/curry.h>
#include <geode/utility/interrupts.h>
#include <geode/utility/Log.h>
#include <geode/utility/const_cast.h>
#include <geode/utility/ProgressIndicator.h>
namespace pentago {

using Log::cout;
using std::endl;

// Given a section s and order o, we define the 4-tensor A(s,o,I) = f(... q[s[k],I[oi[k]]] ...), where f(q0,q1,q2,q3) is the superoutcome for the board with
// quadrants q0,...,q3, q[c,i] is the ith quadrant with counts c, and oi is the inverse of the permutation o.  Hmm, we may have a serious problem.  If we
// apply a reflection transform to standardize a section, the resulting per quadrant reflections will screw up the ordering of the quadrants and break the
// block structure of our file format.  This doesn't apply to a global rotation standardization because we can cancel the global rotation with local rotations.
// This may force us to not standardize via reflections.  Let's see how bad that is: yeah, we lose almost a factor of two if we ignore reflections.  Extremely
// lame.  There is a two pass method of generating a reflected file from an unreflected file, by applying the reflection permutation two quadrant dimensions
// at a time, but this may not be all that much better than simply computing reflected files from scratch.

// Well, so be it.  I don't see any easy way around this, so for now we lose a factor of two.  Let's proceed.  We need a block section of a section A(s,o),
// but all we have is A(gs) for some global rotation g.  Adjust g to a global+local symmetry so that each quadrant moves without rotating.  Let g map quadrant
// k to p[k].  Then we have
//   A(s,o,I) = f(... q[s[k],I[oi[k]]] ...) = gi f(g(... q[s[k],I[oi[k]]] ...)) = gi f(... q[gs[k],I[oi[pi[k]]]] ...) = gi A(gs, 1, I.oi.pi)

// Update: I figured out the easy way around this.  Rotation minimal quadrants are now ordered so that reflected pairs are adjacent in the ordering, and all
// quadrants which aren't fixed by reflection come first.  By enforcing an even block size, we ensure that reflections never cross blocks.  Unfortunately,
// after a board is reflected each quadrant must be rotated to restore minimality, and this rotation can be different per quadrant.  Thus we lose the
// simplicity of applying the same symmetry to each element.  This is a bit complicated (I hate Z_2).  Again, we seek A(s,o), and we have A(gs) for a global
// symmetry g factored as g = rw for a reflection r and rotation w, where w moves each quadrant without rotating.  Factor the reflection r as r = rl rq = rq rl,
// where rl reflects each quadrant locally and rq swaps quadrants 0 and 3.  We have
//   A(s,o,I) = f(... q[s[k],I[oi[k]]] ...) = gi f(g(... q[s[k],I[oi[k]]] ...)) = gi f(r(... q[ws[k],I[oi[wpi[k]]]] ...)) = gi f(rl(... q[gs[k],I[oi[pi[k]]]] ...))
// rl does not preserve rotation minimality of each quadrant, so define maps rs and ri s.t. rl[q[c,i]] = rs[c,i]q[c,ri[c,i]].  Let I.oi.pi = Iop.  Then
//   A(s,o,I) = gi f(... rs[gs[k],Iop[k]] q[gs[k],ri[gs[k],Iop[k]]] ...) = wi rq (... rs[gs[k],Iop[k]] ...) f(... q[gs[k],ri[gs[k],Iop[k]]] ...)

static void endgame_verify_board(const char* prefix, const board_t board, const Vector<super_t,2>& result, bool verbose) {
  const bool turn = popcount(unpack(board,0))!=popcount(unpack(board,1));
  const board_t flipped = turn?pack(unpack(board,1),unpack(board,0)):board;
  const super_t win = super_evaluate_all(true,36,flipped);
  const super_t loss = ~super_evaluate_all(false,36,flipped);
  if (result[turn]!=win || result[1-turn]!=loss)
    for (int r=0;r<256;r++) {
      int fast = result[turn](r)-result[1-turn](r);
      int slow = win(r)-loss(r);
      if (fast!=slow) {
        if (verbose)
          cout << str_board(transform_board(symmetry_t(0,r),board)) << endl;
        section_t section;
        for (int i=0;i<4;i++)
          section.counts[i] = count(quadrant(board,i));
        THROW(RuntimeError,"%s failed: section %s, turn %d, board %lld, rotation %d, fast %d, slow %d",prefix,str(section),turn,board,r,fast,slow);
      }
    }
}

static void endgame_sparse_verify(RawArray<const board_t> boards, RawArray<const Vector<super_t,2>> wins, Random& random, int samples) {
  GEODE_ASSERT(boards.size()==wins.size());
  GEODE_ASSERT((unsigned)samples<=(unsigned)boards.size());
  // Check samples in random order
  Array<int> permutation = arange(boards.size()).copy();
  ProgressIndicator progress(samples,true);
  for (int i=0;i<samples;i++) {
    swap(permutation[i],permutation[random.uniform<int>(i,boards.size())]);
    endgame_verify_board("endgame sparse verify",boards[permutation[i]],wins[permutation[i]],true);
    progress.progress();
  }
}

}
using namespace pentago;

void wrap_verify() {
  GEODE_FUNCTION(endgame_sparse_verify)
}
