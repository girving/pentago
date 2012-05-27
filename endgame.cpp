// Endgame database computation

#include "symmetry.h"
#include "superscore.h"
#include "all_boards.h"
#include <other/core/array/NdArray.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/python/module.h>
#include <other/core/utility/interrupts.h>
#include <other/core/utility/Log.h>
namespace pentago {

using Log::cout;
using std::endl;

static void final_endgame_slice(section_t section, Vector<int,4> offset, NdArray<Vector<super_t,2>> results) {
  // Check input consistency
  OTHER_ASSERT(results.rank()==4);
  OTHER_ASSERT(section.valid() && section.sum()==36);
  RawArray<const quadrant_t> all_rmin[4] = {rotation_minimal_quadrants(section.counts[0]),
                                            rotation_minimal_quadrants(section.counts[1]),
                                            rotation_minimal_quadrants(section.counts[2]),
                                            rotation_minimal_quadrants(section.counts[3])};
  Vector<int,4> shape(results.shape.subset(vec(0,1,2,3)));
  for (int i=0;i<4;i++)
    OTHER_ASSERT((unsigned)offset[i]<=(unsigned)all_rmin[i].size() && offset[i]+shape[i]<=all_rmin[i].size());
  if (!shape.product())
    return;
  RawArray<const quadrant_t> rmin[4] = {all_rmin[0].slice(offset[0],offset[0]+shape[0]),
                                        all_rmin[1].slice(offset[1],offset[1]+shape[1]),
                                        all_rmin[2].slice(offset[2],offset[2]+shape[2]),
                                        all_rmin[3].slice(offset[3],offset[3]+shape[3])};

  // Run the outer three dimensional loops in parallel
  const int first_three = shape[0]*shape[1]*shape[2];
  #pragma omp parallel for
  for (int ijk=0;ijk<first_three;ijk++) {
    if (interrupted())
      continue;
    const int ij = ijk/shape[2],
              k = ijk-ij*shape[2],
              i = ij/shape[1],
              j = ij-i*shape[1];
    const quadrant_t q0 = rmin[0][i],
                     q1 = rmin[1][j],
                     q2 = rmin[2][k],
                     s0b = unpack(q0,0),
                     s0w = unpack(q0,1),
                     s1b = unpack(q1,0),
                     s1w = unpack(q1,1),
                     s2b = unpack(q2,0),
                     s2w = unpack(q2,1);
    const side_t s012b = quadrants(s0b,s1b,s2b,0),
                 s012w = quadrants(s0w,s1w,s2w,0);
    RawArray<Vector<super_t,2>> slice(shape[3],&results(i,j,k,0));
    // Do the inner loop in serial to avoid overhead
    for (int l=0;l<shape[3];l++) {
      const quadrant_t q3 = rmin[3][l],
                       s3b = unpack(q3,0),
                       s3w = unpack(q3,1);
      const side_t side_black = s012b|(side_t)s3b<<3*16,
                   side_white = s012w|(side_t)s3w<<3*16;
      const super_t wins_black = super_wins(side_black),
                    wins_white = super_wins(side_white);
      slice[l][0] = wins_black&~wins_white;
      slice[l][1] = wins_white&~wins_black;
    }
  }
}

}
using namespace pentago;

void wrap_endgame() {
  OTHER_FUNCTION(final_endgame_slice)
}
