// Midgame solver in metal

#include <metal_stdlib>
#include "pentago/mid/subsets.h"
#include "pentago/mid/internal.h"
#include "pentago/base/gen/halfsuper_wins.h"
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

kernel void wins(constant info_t* I [[buffer(0)]],
                 device halfsuper_t* all_wins [[buffer(1)]],
                 const uint index [[thread_position_in_grid]]) {
  SEARCH(s, wins_offsets)
  all_wins[index] = mid_wins(*I, n, s);
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

// iOS has an unfortunate 256 MB buffer size limit, so we need to split workspace into up to four buffers
constant int chunk_size = (uint64_t(256) << 20) / sizeof(halfsupers_t);
constant int chunk_bits = 23;
static_assert(chunk_size == 1 << chunk_bits, "");

struct workspace_t {
  METAL_DEVICE halfsupers_t* chunks[4];
};

struct workspace_io_t {
  workspace_t w;
  int offset, stride;
  
  METAL_DEVICE halfsupers_t& operator()(const int i, const int j) const {
    const int r = i * stride + j + offset;
    return w.chunks[r >> chunk_bits][r & (chunk_size - 1)];
  }
};

workspace_io_t slice(const workspace_t w, const grab_t g) {
  return workspace_io_t{w, g.lo, g.ny};
}

kernel void inner(constant inner_t* I [[buffer(0)]],
                  constant uint16_t* cs1ps [[buffer(1)]],
                  constant set_t* sets1p [[buffer(2)]],
                  constant halfsuper_t* wins [[buffer(3)]],
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
                 wins + I->wins_offset, results, w, I0[I->sets0_offset + s0], s1p);
}
