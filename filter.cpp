// Multidimensional superscore filtering to precondition zlib compression

#include <pentago/filter.h>
#include <pentago/superscore.h>
#include <pentago/utility/aligned.h>
#include <pentago/utility/char_view.h>
#include <other/core/array/Array2d.h>
#include <other/core/array/Array4d.h>
#include <other/core/array/NdArray.h>
#include <other/core/math/sse.h>
#include <other/core/python/module.h>
namespace pentago {

using std::cout;
using std::endl;

static inline uint8_t binary(bool b0, bool b1, bool b2, bool b3, bool b4, bool b5, bool b6, bool b7) {
  return b0|b1<<1|b2<<2|b3<<3|b4<<4|b5<<5|b6<<6|b7<<7;
}

static inline int ternary(int a0, int a1, int a2, int a3) {
  return a0+3*(a1+3*(a2+3*a3));
}

static inline int ternary(int a0, int a1, int a2, int a3, int a4, int a5, int a6, int a7) {
  return a0+3*(a1+3*(a2+3*(a3+3*(a4+3*(a5+3*(a6+3*a7))))));
}

static inline int val(const Vector<super_t,2>& s, int r0, int r1, int r2, int r3) {
  return s[1](r0,r1,r2,r3)-s[0](r0,r1,r2,r3)+1;
}

static inline void set_val(Vector<super_t,2>& s, int r0, int r1, int r2, int r3, int new_) {
  int old = val(s,r0,r1,r2,r3);
  super_t bit = super_t::singleton(r0,r1,r2,r3);
  if ((old<0)!=(new_<0))
    s[0] ^= bit;
  if ((old>0)!=(new_>0))
    s[1] ^= bit;
}

static Array<Vector<int,3>> count_outer_causal_cases(Array<const Vector<super_t,2>,4> data) {
  Array<Vector<int,3>> counts(81);
  const Vector<int,4> shape = data.shape;
  for (int i0=1;i0<shape[0];i0++)
  for (int i1=1;i1<shape[1];i1++)
  for (int i2=1;i2<shape[2];i2++)
  for (int i3=1;i3<shape[3];i3++)
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++)
    for (int r3=1;r3<4;r3++)
      counts[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                     val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                     val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                     val(data(i0,i1,i2,i3-1),r0,r1,r2,r3))][val(data(i0,i1,i2,i3),r0,r1,r2,r3)]++;
  return counts;
}

static Array<Vector<int,3>> count_causal_cases(Array<const Vector<super_t,2>,4> data) {
  Array<Vector<int,3>> counts(6561);
  const Vector<int,4> shape = data.shape;
  for (int i0=1;i0<shape[0];i0++)
  for (int i1=1;i1<shape[1];i1++)
  for (int i2=1;i2<shape[2];i2++)
  for (int i3=1;i3<shape[3];i3++)
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++)
    for (int r3=1;r3<4;r3++)
      counts[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                     val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                     val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                     val(data(i0,i1,i2,i3-1),r0,r1,r2,r3),
                     val(data(i0,i1,i2,i3),r0-1,r1,r2,r3),
                     val(data(i0,i1,i2,i3),r0,r1-1,r2,r3),
                     val(data(i0,i1,i2,i3),r0,r1,r2-1,r3),
                     val(data(i0,i1,i2,i3),r0,r1,r2,r3-1))][val(data(i0,i1,i2,i3),r0,r1,r2,r3)]++;
  return counts;
}

static void outer_causal_filter(Array<const Vector<int,3>> table, Array<Vector<super_t,2>,4> data) {
  OTHER_ASSERT(table.size()==81);
  const Vector<int,4> shape = data.shape;
  for (int i0=shape[0]-1;i0>0;i0--)
  for (int i1=shape[1]-1;i1>0;i1--)
  for (int i2=shape[2]-1;i2>0;i2--)
  for (int i3=shape[3]-1;i3>0;i3--) {
    auto& s = data(i0,i1,i2,i3);
    for (int r0=3;r0>0;r0--) 
    for (int r1=3;r1>0;r1--) 
    for (int r2=3;r2>0;r2--) 
    for (int r3=3;r3>0;r3--)
      set_val(s,r0,r1,r2,r3,table[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                                          val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                                          val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                                          val(data(i0,i1,i2,i3-1),r0,r1,r2,r3))][val(s,r0,r1,r2,r3)]);
  }
}

static void inverse_outer_causal_filter(Array<const Vector<int,3>> table, Array<Vector<super_t,2>,4> data) {
  OTHER_ASSERT(table.size()==81);
  const Vector<int,4> shape = data.shape;
  for (int i0=1;i0<shape[0];i0++)
  for (int i1=1;i1<shape[1];i1++)
  for (int i2=1;i2<shape[2];i2++)
  for (int i3=1;i3<shape[3];i3++) {
    auto& s = data(i0,i1,i2,i3);
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++)
    for (int r3=1;r3<4;r3++) {
      const Vector<int,3>& t = table[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                                             val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                                             val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                                             val(data(i0,i1,i2,i3-1),r0,r1,r2,r3))];
      set_val(s,r0,r1,r2,r3,t.find(val(s,r0,r1,r2,r3)));
    }
  }
}

static void arbitrary_causal_filter(Array<const Vector<int,3>> table, Array<Vector<super_t,2>,4> data) {
  OTHER_ASSERT(table.size()==6561);
  const Vector<int,4> shape = data.shape;
  for (int i0=shape[0]-1;i0>0;i0--)
  for (int i1=shape[1]-1;i1>0;i1--)
  for (int i2=shape[2]-1;i2>0;i2--)
  for (int i3=shape[3]-1;i3>0;i3--) {
    auto& s = data(i0,i1,i2,i3);
    for (int r0=3;r0>0;r0--) 
    for (int r1=3;r1>0;r1--) 
    for (int r2=3;r2>0;r2--) 
    for (int r3=3;r3>0;r3--)
      set_val(s,r0,r1,r2,r3,table[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                                          val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                                          val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                                          val(data(i0,i1,i2,i3-1),r0,r1,r2,r3),
                                          val(data(i0,i1,i2,i3),r0-1,r1,r2,r3),
                                          val(data(i0,i1,i2,i3),r0,r1-1,r2,r3),
                                          val(data(i0,i1,i2,i3),r0,r1,r2-1,r3),
                                          val(data(i0,i1,i2,i3),r0,r1,r2,r3-1))][val(s,r0,r1,r2,r3)]);
  }
}

static void inverse_arbitrary_causal_filter(Array<const Vector<int,3>> table, Array<Vector<super_t,2>,4> data) {
  OTHER_ASSERT(table.size()==6561);
  const Vector<int,4> shape = data.shape;
  for (int i0=1;i0<shape[0];i0++)
  for (int i1=1;i1<shape[1];i1++)
  for (int i2=1;i2<shape[2];i2++)
  for (int i3=1;i3<shape[3];i3++) {
    auto& s = data(i0,i1,i2,i3);
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++)
    for (int r3=1;r3<4;r3++) {
      const Vector<int,3>& t = table[ternary(val(data(i0-1,i1,i2,i3),r0,r1,r2,r3),
                                             val(data(i0,i1-1,i2,i3),r0,r1,r2,r3),
                                             val(data(i0,i1,i2-1,i3),r0,r1,r2,r3),
                                             val(data(i0,i1,i2,i3-1),r0,r1,r2,r3),
                                             val(data(i0,i1,i2,i3),r0-1,r1,r2,r3),
                                             val(data(i0,i1,i2,i3),r0,r1-1,r2,r3),
                                             val(data(i0,i1,i2,i3),r0,r1,r2-1,r3),
                                             val(data(i0,i1,i2,i3),r0,r1,r2,r3-1))];
      set_val(s,r0,r1,r2,r3,t.find(val(s,r0,r1,r2,r3)));
    }
  }
}

static Array<int,2> count_rotation_cases(Array<const Vector<super_t,2>,4> data) {
  Array<int,2> counts(4,81);
  for (auto& s : data.flat) {
    for (int r0=1;r0<4;r0++)
    for (int r1=1;r1<4;r1++)
    for (int r2=1;r2<4;r2++) {
      counts(0,ternary(val(s,0,r0,r1,r2),val(s,1,r0,r1,r2),val(s,2,r0,r1,r2),val(s,3,r0,r1,r2)))++;
      counts(1,ternary(val(s,r0,0,r1,r2),val(s,r0,1,r1,r2),val(s,r0,2,r1,r2),val(s,r0,3,r1,r2)))++;
      counts(2,ternary(val(s,r0,r1,0,r2),val(s,r0,r1,1,r2),val(s,r0,r1,2,r2),val(s,r0,r1,3,r2)))++;
      counts(3,ternary(val(s,r0,r1,r2,0),val(s,r0,r1,r2,1),val(s,r0,r1,r2,2),val(s,r0,r1,r2,3)))++;
    }
  }
  return counts;
}

void interleave(RawArray<Vector<super_t,2>> data) {
  for (auto& s : data)
    s = interleave_super(s);
}

void uninterleave(RawArray<Vector<super_t,2>> data) {
  for (auto& s : data)
    s = uninterleave_super(s);
}

// Turn 5*256 win/loss/tie values into 256 bytes
static inline void compact_chunk(uint8_t dst[256], const Vector<super_t,2> src[5]) {
  const super_t s[2][5] = {{src[0].x,src[1].x,src[2].x,src[3].x,src[4].x},
                           {src[0].y,src[1].y,src[2].y,src[3].y,src[4].y}};
  const uint8_t* c0 = (const uint8_t*)s[0]; // size 5*32
  const uint8_t* c1 = (const uint8_t*)s[1]; // size 5*32
  #define GET_FIVE(c,i) ({ \
    const int bit = 5*i, lo = bit&7, hi = bit>>3; \
    const uint8_t lo_mask = (1<<min(5,8-lo))-1, \
                  hi_mask = 0x1f&~lo_mask; \
    (c[hi]>>lo&lo_mask) | (lo>3?c[hi+1]<<(8-lo)&hi_mask:0); \
    })
  for (int i=0;i<256;i++)
    dst[i] = pack_table[GET_FIVE(c0,i)]+2*pack_table[GET_FIVE(c1,i)];
  #undef GET_FIVE
}

// Inverse of compact_chunk
static inline void uncompact_chunk(Vector<super_t,2> dst[5], const uint8_t src[256]) {
  super_t d[2][5];
  memset(d,0,sizeof(d));
  uint8_t* c0 = (uint8_t*)d[0]; // size 5*32
  uint8_t* c1 = (uint8_t*)d[1]; // size 5*32
  #define SET_FIVE(c,i,v) ({ \
    const int bit = 5*i, lo = bit&7, hi = bit>>3; \
    c[hi] |= v<<lo; \
    if (lo>3)  c[hi+1] |= v>>(8-lo); \
    })
  for (int i=0;i<256;i++) {
    SET_FIVE(c0,i,unpack_table[src[i]][0]);
    SET_FIVE(c1,i,unpack_table[src[i]][1]);
  }
  #undef SET_FIVE
  for (int i=0;i<5;i++) {
    dst[i].x = d[0][i];
    dst[i].y = d[1][i];
  }
}

// Compaction packs 5 win/loss/tie values into 1 bytes (3**5 < 2**8), destroying the original array
Array<uint8_t> compact(Array<Vector<super_t,2>> src) {
  Array<uint8_t> dst = char_view_own(src).slice_own(0,(256*src.size()+4)/5);
  const int chunks = src.size()/5,
            extra = src.size()-5*chunks;
  for (int i=0;i<chunks;i++)
    compact_chunk(&dst[256*i],&src[5*i]);
  if (extra) {
    Vector<super_t,2> rest[5];
    memset(rest,0,sizeof(rest));
    memcpy(rest,src.data()+5*chunks,sizeof(Vector<super_t,2>)*extra);
    compact_chunk((uint8_t*)rest,rest);
    memcpy(dst.data()+256*chunks,rest,dst.size()-256*chunks);
  }
  return dst;
}

// Inverse of compact.  Not in-place since the output is larger than the input.
Array<Vector<super_t,2>> uncompact(Array<const uint8_t> src) {
  const auto dst = aligned_buffer<Vector<super_t,2>>(5*src.size()/256);
  OTHER_ASSERT(src.size()==(256*dst.size()+4)/5);
  const int chunks = dst.size()/5, 
            extra = dst.size()-5*chunks;
  for (int i=0;i<chunks;i++)
    uncompact_chunk(&dst[5*i],&src[256*i]);
  if (extra) {
    uint8_t src_rest[256];
    memset(src_rest,0,sizeof(src_rest));
    memcpy(src_rest,src.data()+256*chunks,src.size()-256*chunks);
    Vector<super_t,2> dst_rest[5];
    uncompact_chunk(dst_rest,src_rest);
    memcpy(dst.data()+5*chunks,dst_rest,sizeof(Vector<super_t,2>)*extra);
  }
  return dst;
}

Array<int,2> count_byte_cases(RawArray<const Vector<super_t,2>> data) {
  auto copy = data.copy();
  interleave(copy);
  auto bytes = char_view(data);
  Array<int,2> cases(256,256);
  for (int i=1;i<bytes.size();i++)
    cases(bytes[i-1],bytes[i])++;
  return cases;
}

typedef Vector<super_t,2> F3;

static inline F3 add(const F3& f, const F3& g) {
  super_t sx = f.x^g.x,
          sy = f.y^g.y^(f.x&g.x);
  sx ^= f.y&g.y;
  super_t o = sx&sy;
  return vec(sx^o,sy^o);
}

static inline F3 neg(const F3& f) {
  return F3(f.y,f.x);
}

static inline F3 sub(const F3& f, const F3& g) {
  return add(f,neg(g));
}

static inline F3 neg_add(const F3& f, const F3& g) {
  return neg(add(f,g));
}

static inline F3 rev_sub(const F3& f, const F3& g) {
  return sub(g,f);
}

template<bool invert> static inline void wavelet(F3& f0, F3& f1, F3& f2, F3& f3, F3& f4, F3& f5, F3& f6, F3& f7) {
  if (!invert) {
    const F3 p0 = add(f0,f1),
             m0 = sub(f0,f1),
             p1 = add(f2,f3),
             m1 = sub(f2,f3),
             p2 = add(f4,f5),
             m2 = sub(f4,f5),
             p3 = add(f6,f7),
             m3 = sub(f6,f7);
    const F3 pp0 = add(p0,p1),
             pm0 = sub(p0,p1),
             pp1 = add(p2,p3),
             pm1 = sub(p2,p3);
    const F3 ppp = add(pp0,pp1),
             ppm = sub(pp0,pp1);
    f0 = ppp;
    f1 = ppm;
    f2 = pm0;
    f3 = pm1;
    f4 = m0;
    f5 = m1;
    f6 = m2;
    f7 = m3;
  } else {
    const F3 ppp = f0,
             ppm = f1,
             pm0 = f2,
             pm1 = f3,
             m0 = f4,
             m1 = f5,
             m2 = f6,
             m3 = f7;
    const F3 pp0 = neg_add(ppp,ppm),
             pp1 = rev_sub(ppp,ppm);
    const F3 p0 = neg_add(pp0,pm0),
             p1 = rev_sub(pp0,pm0),
             p2 = neg_add(pp1,pm1),
             p3 = rev_sub(pp1,pm1);
    f0 = neg_add(p0,m0);
    f1 = rev_sub(p0,m0);
    f2 = neg_add(p1,m1);
    f3 = rev_sub(p1,m1);
    f4 = neg_add(p2,m2);
    f5 = rev_sub(p2,m2);
    f6 = neg_add(p3,m3);
    f7 = rev_sub(p3,m3);
  }
}

template<bool invert> static void wavelet_loop(RawArray<F3> data, const Vector<int,4> shape, const Vector<int,4> strides) {
  // Size 8 wavelet transform
  #define WAVELET8() \
    wavelet<invert>(data[base+stride*0],data[base+stride*1],data[base+stride*2],data[base+stride*3], \
                    data[base+stride*4],data[base+stride*5],data[base+stride*6],data[base+stride*7]);
  // Loop over three dimensions and wavelet transform the fourth
  const int stride = strides[3];
  #define LOOP(n) \
    for (int i0=0;i0<shape[0];i0++) \
      for (int i1=0;i1<shape[1];i1++) \
        for (int i2=0;i2<shape[2];i2++) { \
          const int base = strides[0]*i0+strides[1]*i1+strides[2]*i2; \
          WAVELET##n(); \
        }
  // For now, we only bother with size 8 wavelet transforms
  switch (shape[3]) {
    case 8: LOOP(8); break;
    default: break;
  }
}

template<bool invert> static void wavelet_helper(RawArray<F3,4> data) {
  const auto shape = data.shape;
  const Vector<int,4> strides(shape[1]*shape[2]*shape[3],shape[2]*shape[3],shape[3],1);
  for (int a=0;a<4;a++) {
    auto a_shape = shape, a_strides = strides;
    swap(a_shape[a],a_shape[3]);
    swap(a_strides[a],a_strides[3]);
    wavelet_loop<invert>(data.flat,a_shape,a_strides);
  }
}

void wavelet_transform(RawArray<F3,4> data) {
  wavelet_helper<false>(data); 
  interleave(data.flat);
}

void wavelet_untransform(RawArray<F3,4> data) {
  uninterleave(data.flat);
  wavelet_helper<true>(data); 
}

}
using namespace pentago;

void wrap_filter() {
  OTHER_FUNCTION(count_causal_cases)
  OTHER_FUNCTION(count_outer_causal_cases)
  OTHER_FUNCTION(count_rotation_cases)
  OTHER_FUNCTION(count_byte_cases)
  OTHER_FUNCTION(interleave)
  OTHER_FUNCTION(uninterleave)
  OTHER_FUNCTION(arbitrary_causal_filter)
  OTHER_FUNCTION(inverse_arbitrary_causal_filter)
  OTHER_FUNCTION(outer_causal_filter)
  OTHER_FUNCTION(inverse_outer_causal_filter)
  OTHER_FUNCTION(compact)
  OTHER_FUNCTION(uncompact)
  OTHER_FUNCTION(wavelet_transform)
  OTHER_FUNCTION(wavelet_untransform)
}
