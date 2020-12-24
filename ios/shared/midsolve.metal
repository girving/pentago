// Midgame solver in metal

#include <metal_stdlib>
#include "../pentago/mid/subsets.h"
#include "../pentago/mid/internal.h"
#include "../pentago/base/gen/halfsuper_wins.h"
using namespace metal;
using namespace pentago;

inline int find_n(const int spots, constant int* offsets, const int i) {
  // Invariant: answer ∈ [lo,hi)
  int lo = 0;
  int hi = spots + 1;
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (i < offsets[mid]) hi = mid;
    else lo = mid;
  }
  return lo;
}
#define SEARCH(i, offsets) \
  const int n = find_n(I->spots, I->offsets, index); \
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
  const auto s0 = s / I->sets1p_size;
  const auto s1p = s - s0 * I->sets1p_size;
  const workspace_t w{{workspace0, workspace1, workspace2, workspace3}};
  pentago::inner(*I, cs1ps + I->cs1ps_offset, sets1p + I->sets1p_offset,
                 wins1 + I->wins1_offset, results, w, I0[I->sets0_offset + s0], s1p);
}

// Partition a loop into chunks based on the total number of threads.  Returns a half open interval.
__attribute__((unused)) static inline int2
partition_loop(const int steps, const int chunks, const int chunk) {
  const int steps_per_chunk = steps / chunks;  // Round down, so some chunks will get one more step
  const int extra_steps = steps % chunks;  // The first extra_steps chunks will get one extra step
  const int start = steps_per_chunk*chunk + min(extra_steps, chunk);
  const int end = start+steps_per_chunk + (chunk<extra_steps);
  return int2(start, end);
}

kernel void set1_info(constant transposed_t* I [[buffer(0)]],
                      device set1_info_t* I1 [[buffer(1)]],
                      const uint index [[thread_position_in_grid]]) {
  SEARCH(s1, sets1_offsets)
  I1[index] = make_set1_info(*I, n, s1);
}

kernel void wins0(constant transposed_t* I [[buffer(0)]],
                  device halfsuper_t* all_wins0 [[buffer(1)]],
                  const uint index [[thread_position_in_grid]]) {
  SEARCH(s0, sets0_offsets)
  const helper_t<constant transposed_t&> H{*I, n};
  all_wins0[index] = halfsuper_wins(H.root0() | side(I->empty, H.sets0(), s0), H.parity());
}

kernel void cs0ps(constant transposed_t* I [[buffer(0)]],
                  device uint16_t* cs0ps [[buffer(1)]],
                  const uint index [[thread_position_in_grid]]) {
  SEARCH(i, cs0ps_offsets)
  const helper_t<constant transposed_t&> H{*I, n};
  const int moves = I->spots - n;
  const int s0p = i / moves;
  const int m = i - s0p * moves;
  cs0ps[index] = make_cs0ps(H.sets0p(), s0p, m);
}

kernel void transposed(constant transposed_inner_t* I [[buffer(0)]],
                       constant set1_info_t* I1s [[buffer(1)]],
                       constant halfsuper_t* all_wins0 [[buffer(2)]],
                       constant uint16_t* all_cs0ps [[buffer(3)]],
                       device halfsupers_t* results [[buffer(4)]],
                       device halfsupers_t* workspace0 [[buffer(5)]],
                       device halfsupers_t* workspace1 [[buffer(6)]],
                       device halfsupers_t* workspace2 [[buffer(7)]],
                       device halfsupers_t* workspace3 [[buffer(8)]],
                       const uint s [[thread_position_in_grid]]) {
  // Various constants
  const int n = I->n;
  const int spots = I->spots;
  const bool done = spots == n;
  const int k0 = n >> 1;
  const int k1 = n - k0;
  const auto sets0p = make_sets(spots-k1, k0);

  // Input and output
  const workspace_t w{{workspace0, workspace1, workspace2, workspace3}};
  const auto input = slice(w, I->input);
  const auto output = slice(w, I->output);
  const auto cs0ps = all_cs0ps + I->cs0ps_offset;

  // Unpack indices
  const auto s1 = s / sets0p.size;
  const auto s0p = s - s1 * sets0p.size;
  constant auto& I1 = I1s[I->sets1_offset + s1];
  const auto c = commute(I1, sets0p, s0p);

  // Consider each move in turn
  halfsupers_t us = {{0}, {0}};
  if (done)
    us.notlose = ~halfsuper_t(0);
  for (int i = 0; i < spots-n; i++)
    us |= input(s1, cs0ps[s0p*(spots-n) + i]);

  // Account for immediate results
  const auto wins0 = all_wins0[I->sets0_offset + c.s0];
  const auto inplay = ~(wins0 | I1.wins1.after);
  us.win = (inplay & us.win) | (wins0 & ~I1.wins1.after);
  us.notlose = (inplay & us.notlose) | wins0;

  // If we're far enough along, remember results
  if (n <= 1)
    results[n + s1] = us;

  // Prepare for the slice above
  output(c.s0, c.s1p) = rmax(~us) | I1.wins1.before;
}
