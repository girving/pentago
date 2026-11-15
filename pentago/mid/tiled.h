// Memory-lean midgame solver: slices tiled by prefix occupancy, tiles stored compressed
//
// midsolve (midengine.h) keeps two full slices of halfsupers dense in memory, which is 4 GB for a
// 16 stone root.  This variant tiles each slice by the occupancy pattern of the k highest empty
// spots.  Because subsets are ranked in colex order, every pattern is a contiguous block of rows and
// of columns, so a tile is a rectangular sub-array and the existing index arithmetic applies
// unchanged.  An output tile depends on at most k+1 input tiles (the same pattern, or the pattern
// with one prefix spot filled by the mover), so we decode only those into dense staging, run the
// inner loop, and encode the result straight from the loop with the codec in codec.h.
//
// Output tiles are grouped by row pattern, which is exactly the set of consumers of each input tile,
// so every input tile is decoded once and freed when its group finishes, keeping about one slice live.
//
// Memory is one arena carved into fixed regions plus a pool of equal sized pages for compressed
// streams, so the solver never allocates per tile and runs unchanged in freestanding WebAssembly.
// Threads are a persistent pool of participants on shared memory: native std::threads, or wasm
// workers calling midsolve_tiled_worker.
#pragma once

#include "pentago/mid/midengine.h"
#include <cstdint>
namespace pentago {

struct tiled_options_t {
  int prefix = 0;        // Number of prefix spots used for tiling (3^prefix tiles per slice); 0 picks by board size
  int threads = 1;       // Participants in the worker pool, including the caller
  bool early_exit = true;  // Stop reading children once the mover's set covers every in-play rotation
  bool merged = false;     // One pass computing win and not-lose together: about 30% faster, 1.4-1.6x the memory
  bool verbose = false;  // Print per slice memory and timing (native only)
};

struct tiled_stats_t {
  int64_t peak_bytes = 0;         // Peak arena bytes in use (tables + staging + pages + bookkeeping)
  int64_t peak_pages = 0;         // Peak bytes of compressed pages in use
  int64_t arena_bytes = 0;        // Total arena reserved
  int64_t entries = 0;            // Total entries encoded
  double seconds = 0;
  double decode_seconds = 0;      // Wall time decoding input tiles
  double compute_seconds = 0;     // Wall time computing and encoding output tiles
};

// Same results as midsolve_internal, using far less memory
Vector<halfsupers_t,1+MID_MAX_SPOTS> midsolve_tiled_internal(const high_board_t root, const tiled_options_t& options,
                                                            tiled_stats_t* stats = nullptr);

#ifndef __wasm__
// Same results as midsolve
mid_values_t midsolve_tiled(const high_board_t board, const tiled_options_t& options, tiled_stats_t* stats = nullptr);
#endif

}  // namespace pentago
