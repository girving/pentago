// Multidimensional superscore filtering to precondition zlib compression

#include "superscore.h"
#include <other/core/array/NdArray.h>
#include <other/core/array/IndirectArray.h>
#include <other/core/python/module.h>
namespace pentago {

using std::cout;
using std::endl;

static inline uint8_t bits(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7) {
  return b0|b1<<1|b2<<2|b3<<3|b4<<4|b5<<5|b6<<6|b7<<7;
}

static Array<Vector<int,2>> count_causal_cases(NdArray<const super_t> data) {
  OTHER_ASSERT(data.rank()==4);
  const Vector<int,4> shape(data.shape.subset(vec(0,1,2,3)));

  // Count everything
  Array<Vector<int,2>> counts(256);
  for (int i0=1;i0<shape[0];i0++)
  for (int i1=1;i1<shape[1];i1++)
  for (int i2=1;i2<shape[2];i2++)
  for (int i3=1;i3<shape[3];i3++)
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++)
    for (int r3=1;r3<4;r3++)
      counts[bits(data(i0-1,i1,i2,i3)(r0,r1,r2,r3),
                  data(i0,i1-1,i2,i3)(r0,r1,r2,r3),
                  data(i0,i1,i2-1,i3)(r0,r1,r2,r3),
                  data(i0,i1,i2,i3-1)(r0,r1,r2,r3),
                  data(i0,i1,i2,i3)(r0-1,r1,r2,r3),
                  data(i0,i1,i2,i3)(r0,r1-1,r2,r3),
                  data(i0,i1,i2,i3)(r0,r1,r2-1,r3),
                  data(i0,i1,i2,i3)(r0,r1,r2,r3-1))][data(i0,i1,i2,i3)(r0,r1,r2,r3)]++;
  return counts;
}

}
using namespace pentago;

void wrap_filter() {
  OTHER_FUNCTION(count_causal_cases)
}
