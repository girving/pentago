// Midgame solver in metal

#include <metal_stdlib>
#include "../pentago/mid/subsets.h"
#include "../pentago/mid/internal.h"
#include "../pentago/base/gen/halfsuper_wins.h"
using namespace metal;
using namespace pentago;

inline int find_n(constant info_t* I, constant int* offsets, const int i) {
  // Invariant: answer ∈ [lo,hi)
  int lo = 0;
  int hi = I->spots + 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (i < offsets[mid]) hi = mid;
    else lo = mid;
  }
  return lo;
}
#define SEARCH(i, offsets) \
  if (index >= uint(I->offsets[I->spots+1])) return; \
  const int n = find_n(I, I->offsets, index); \
  const int i = index - I->offsets[n];

kernel void sets1p(constant info_t* I [[buffer(0)]],
                   device set_t* sets1p [[buffer(1)]],
                   const uint index [[thread_position_in_grid]]) {
  SEARCH(s, sets1p_offsets)
  const helper_t<> H{*I, n};
  sets1p[index] = get(H.sets1p(), s);
}

kernel void wins1(constant info_t* I [[buffer(0)]],
                  device wins1_t* all_wins1 [[buffer(1)]],
                  const uint index [[thread_position_in_grid]]) {
  SEARCH(s, wins1_offsets)
  all_wins1[index] = mid_wins1(*I, n, s);
}

kernel void cs1ps(constant info_t* I [[buffer(0)]],
                  constant set_t* sets1p [[buffer(1)]],
                  device uint16_t* cs1ps [[buffer(2)]],
                  const uint index [[thread_position_in_grid]]) {
  SEARCH(i, cs1ps_offsets)
  const auto sets1p_n = sets1p + I->sets1p_offsets[n];
  cs1ps[index] = make_cs1ps(*I, sets1p_n, n, i);
}

kernel void set0_info(constant info_t* I [[buffer(0)]],
                      device set0_info_t* I0 [[buffer(1)]],
                      const uint index [[thread_position_in_grid]]) {
  SEARCH(s0, sets0_offsets)
  I0[index] = make_set0_info(*I, n, s0);
}

kernel void inner(constant inner_t* I [[buffer(0)]],
                  constant uint16_t* cs1ps [[buffer(1)]],
                  constant set_t* sets1p [[buffer(2)]],
                  constant wins1_t* wins1 [[buffer(3)]],
                  constant set0_info_t* I0 [[buffer(4)]],
                  device halfsupers_t* results [[buffer(5)]],
                  device halfsupers_t* workspace0 [[buffer(6)]],
                  device halfsupers_t* workspace1 [[buffer(7)]],
                  device halfsupers_t* workspace2 [[buffer(8)]],
                  device halfsupers_t* workspace3 [[buffer(9)]],
                  const uint s [[thread_position_in_grid]]) {
  if (s >= uint(I->output.size)) return;
  const auto s0 = s / I->sets1p_size;
  const auto s1p = s - s0 * I->sets1p_size;
  const workspace_t w{{workspace0, workspace1, workspace2, workspace3}};
  pentago::inner(*I, cs1ps + I->cs1ps_offset, sets1p + I->sets1p_offset,
                 wins1 + I->wins1_offset, results, w, I0[I->sets0_offset + s0], s1p);
}
