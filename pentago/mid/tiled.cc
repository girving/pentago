// Memory-lean midgame solver: slices tiled by prefix occupancy, tiles stored compressed

#include "pentago/mid/tiled.h"
#include "pentago/mid/codec.h"
#include "pentago/mid/internal.h"
#include "pentago/utility/debug.h"
#include <algorithm>
#include <cstring>
#ifdef __wasm__
#include "pentago/utility/wasm_alloc.h"
#else
#include "pentago/utility/log.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#endif
namespace pentago {

using std::max;
using std::min;

namespace {

// ---------------------------------------------------------------------------------------------
// Atomics and a persistent worker pool on shared memory.  Participants (the caller plus threads-1
// workers) grab work items with an atomic counter.  Native workers are std::threads; wasm workers
// are web workers sharing the module's memory and calling midsolve_tiled_worker.

static inline int aload(const int* p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void astore(int* p, const int v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }
static inline int aadd(int* p, const int v) { return __atomic_fetch_add(p, v, __ATOMIC_ACQ_REL); }

// Block until *p != v (wasm with threads) or spin with yields (native).  A wasm build without the
// atomics feature has no workers, so it never waits.
static inline void wait_while(int* p, const int v) {
#if defined(__wasm_atomics__)
  while (aload(p) == v)
    __builtin_wasm_memory_atomic_wait32(p, v, -1);
#elif defined(__wasm__)
  (void)p; (void)v;
#else
  for (int spins = 0; aload(p) == v; spins++)
    if (spins > 100) std::this_thread::yield();
#endif
}
static inline void wake(int* p) {
#if defined(__wasm_atomics__)
  __builtin_wasm_memory_atomic_notify(p, 0x7fffffff);
#else
  (void)p;
#endif
}

struct pool_t {
  int threads = 1;  // Participants including the caller
  int generation = 0, next = 0, count = 0, done = 0, stop = 0, nworkers = 0;
  void (*fn)(void*, int, int) = nullptr;
  void* ctx = nullptr;

  // Work items also learn which participant runs them (0 is the caller), for per-thread scratch
  void work(const int me) {
    for (;;) {
      const int i = aadd(&next, 1);
      if (i >= count) return;
      fn(ctx, i, me);
    }
  }

  // Run f(ctx, i, participant) for i in [0,n) across all participants, returning when every item is done
  void run(const int n, void (*f)(void*, int, int), void* c) {
    if (threads <= 1 || n <= 1) {
      for (int i = 0; i < n; i++) f(c, i, 0);
      return;
    }
    // Workers read the generation before registering, so once all have registered none can miss the
    // bump below.  Without this a worker starting late waits for the following generation and this
    // run never completes, which is what happened on slow CI machines.
    for (int w; (w = aload(&nworkers)) < threads - 1;)
      wait_while(&nworkers, w);
    fn = f; ctx = c;
    astore(&count, n);
    astore(&next, 0);
    astore(&done, 0);
    aadd(&generation, 1);
    wake(&generation);
    work(0);
    aadd(&done, 1);
    for (int d; (d = aload(&done)) < threads;)
      wait_while(&done, d);
  }

  // Worker loop: returns only when stop is set
  void worker() {
    int gen = aload(&generation);  // Before registering, see run()
    const int me = aadd(&nworkers, 1) + 1;
    wake(&nworkers);
    for (;;) {
      if (aload(&stop)) return;  // Stop may have preceded the generation we first saw
      wait_while(&generation, gen);
      gen = aload(&generation);
      if (aload(&stop)) return;
      work(me);
      aadd(&done, 1);
      wake(&done);
    }
  }
};

static pool_t pool;

// ---------------------------------------------------------------------------------------------
// Memory: one arena of fixed regions, plus equal sized pages for compressed streams.

struct arena_t {
  uint8_t* base = nullptr;
  size_t size = 0, used = 0;

  void reserve(const size_t bytes) {
    base = static_cast<uint8_t*>(malloc(bytes));
    GEODE_ASSERT(base);
    size = bytes;
    used = 0;
  }
  void release() {
#ifndef __wasm__
    free(base);
#endif
    base = nullptr;
  }
  template<class T> T* alloc(const size_t n) {
    used = (used + 63) & ~size_t(63);
    T* p = reinterpret_cast<T*>(base + used);
    used += n * sizeof(T);
    GEODE_ASSERT(used <= size);
    return p;
  }
};

struct pages_t {
  static const int page_words = 512, block_pages = 4096;   // 2 KB pages in 8 MB blocks
  static const int max_blocks = 512;                       // 4 GB of pages at most
  uint32_t* blocks[max_blocks];
  int nblocks = 0, capacity = 0;
  // Free pages form an intrusive stack through their first word, so the free list costs no memory.
  // Pops are concurrent inside parallel phases; pushes happen only between phases (the caller frees
  // consumed tiles) or under the growth lock with pages nobody has seen, so there is no ABA hazard.
  int head = -1;              // First free page, or -1
  int nfree = 0;
  int lock = 0;               // Serializes growth inside parallel phases
  int peak_used = 0;

  uint32_t* page(const int i) const { return blocks[i >> 12] + (i & 4095) * page_words; }
  int used() const { return capacity - aload(&nfree); }

  // Add a block of pages.  Only ever runs on one thread at a time.
  void grow() {
    GEODE_ASSERT(nblocks < max_blocks);
    blocks[nblocks++] = static_cast<uint32_t*>(malloc(size_t(block_pages) * page_words * 4));
    const int base = capacity;
    for (int i = 0; i < block_pages - 1; i++) page(base + i)[0] = uint32_t(base + i + 1);
    page(base + block_pages - 1)[0] = uint32_t(aload(&head));
    capacity += block_pages;
    aadd(&nfree, block_pages);
    astore(&head, base);  // Publish after the chain is written
  }

  // Grab a page inside a parallel phase, growing on demand so memory tracks actual use
  int alloc() {
    for (;;) {
      int h = aload(&head);
      if (h >= 0) {
        const int next = int(page(h)[0]);
        if (__atomic_compare_exchange_n(&head, &h, next, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
          aadd(&nfree, -1);
          return h;
        }
        continue;
      }
      // Out of pages: one thread grows while the others wait
      while (__atomic_exchange_n(&lock, 1, __ATOMIC_ACQUIRE)) {}
      if (aload(&head) < 0) grow();
      astore(&lock, 0);
    }
  }
  // Return a chain of pages, caller only
  void release(int first) {
    while (first >= 0) {
      const int next = int(page(first)[0]);
      page(first)[0] = uint32_t(head);
      head = first;
      nfree++;
      first = next;
    }
  }
  // Note the high water mark, caller only
  void ensure(const int n) {
    peak_used = max(peak_used, used());
    while (nfree < n) grow();
  }
  void destroy() {
#ifndef __wasm__
    for (int b = 0; b < nblocks; b++) free(blocks[b]);
#endif
  }
};

// Codec sinks and sources over page chains
struct page_writer_t {
  pages_t* P;
  int first = -1, cur = -1, pos = pages_t::page_words;
  void start() { first = cur = P->alloc(); P->page(cur)[0] = uint32_t(-1); pos = 1; }
  static void emit(void* ctx, const uint32_t word) {
    auto& w = *static_cast<page_writer_t*>(ctx);
    if (w.pos == pages_t::page_words) {
      const int n = w.P->alloc();
      w.P->page(n)[0] = uint32_t(-1);
      w.P->page(w.cur)[0] = uint32_t(n);
      w.cur = n;
      w.pos = 1;
    }
    w.P->page(w.cur)[w.pos++] = word;
  }
};
struct page_reader_t {
  const pages_t* P;
  int cur;
  int pos = 1;
  static uint32_t fetch(void* ctx) {
    auto& r = *static_cast<page_reader_t*>(ctx);
    if (r.cur < 0) return 0;  // Reading past the end yields zeros
    if (r.pos == pages_t::page_words) {
      r.cur = int(r.P->page(r.cur)[0]);
      r.pos = 1;
      if (r.cur < 0) return 0;
    }
    return r.P->page(r.cur)[r.pos++];
  }
};

// ---------------------------------------------------------------------------------------------
// Tile geometry.  Patterns are bitmasks over the k prefix spots, bit p standing for spot S-k+p.
// prow is the pattern of the row player's set (set1 at slice n), pcol of the column player's (set0).

struct geom_t {
  bool valid = false;
  int q1 = 0, q0 = 0;              // Stones of each player among the low (non-prefix) spots
  int rowbase = 0, nrows = 0;      // Row block: absolute row indices [rowbase, rowbase+nrows)
  int colbase = 0, ncols = 0;      // Column block, relative to any row of the block
  int64_t entries() const { return int64_t(nrows) * ncols; }
};

// Sum over pattern bits of choose(f(p), q + index + 1), the contribution of prefix elements to a colex rank
template<class F> static int pattern_rank(const int pattern, const int q, F&& f) {
  int rank = 0, j = q;
  for (int p = 0; p < 32; p++)
    if (pattern >> p & 1)
      rank += choose(f(p), ++j);
  return rank;
}

static geom_t make_geom(const int S, const int k, const int n, const int prow, const int pcol) {
  geom_t g;
  const int k0 = n >> 1, k1 = n - k0, low = S - k;
  if (prow & pcol) return g;
  g.q1 = k1 - __builtin_popcount(prow);
  g.q0 = k0 - __builtin_popcount(pcol);
  if (g.q1 < 0 || g.q0 < 0 || g.q1 > low || g.q0 > low - g.q1) return g;
  const int q0 = g.q0, q1 = g.q1;
  const auto below = [](const int pattern, const int p) { return __builtin_popcount(pattern & ((1 << p) - 1)); };
  g.rowbase = pattern_rank(prow, q1, [=](int p) { return low + p; });
  g.nrows = choose(low, q1);
  g.colbase = pattern_rank(pcol, q0, [=](int p) { return low + p - q1 - below(prow, p); });
  g.ncols = choose(low - q1, q0);
  g.valid = g.entries() > 0;
  return g;
}

// Chunking of a tile's rows, independent of thread count so that compressed sizes are deterministic.
// Small chunks keep even small tiles parallel; each costs a cache warmup and a partial page.
static int chunk_count(const geom_t& g) {
  const int64_t target = 8192;
  return int(max(int64_t(1), min(int64_t(g.nrows), (g.entries() + target - 1) / target)));
}
static int chunk_lo(const geom_t& g, const int c, const int nchunks) { return g.rowbase + int(int64_t(g.nrows) * c / nchunks); }

// Walk a row's columns in colex order: consecutive k-subsets of [0,n) as bitmasks (Gosper's hack)
static inline uint32_t next_subset(const uint32_t mask) {
  const uint32_t low = mask & -mask, ripple = mask + low;
  return ripple | ((mask ^ ripple) >> (__builtin_ctz(low) + 2));
}

// The absolute index of the mover's set along a row, updated incrementally: a Gosper step moves the
// top of the lowest run of elements up by one and packs the rest of the run to the bottom, so only
// those elements' contributions change.  Used by the decoder, which needs nothing else per column.
struct cursor_t {
  int k;                          // Elements (k0)
  uint8_t r[MID_MAX_SPOTS];       // Relative positions, ascending
  uint32_t mask = 0, s0 = 0;

  void start(const set1_info_t& I1, const int k_, const uint32_t mask_) {
    k = k_; mask = mask_; s0 = 0;
    for (uint32_t bits = mask, j = 0; bits; bits &= bits - 1, j++) {
      r[j] = __builtin_ctz(bits);
      s0 += fast_choose(I1.empty0[r[j]], j+1);
    }
  }
  void step(const set1_info_t& I1) {
    if (!k) return;
    const int t = __builtin_ctz(~(mask >> r[0]));  // Length of the lowest run
    for (int j = 0; j < t; j++) s0 -= fast_choose(I1.empty0[r[j]], j+1);
    for (int j = 0; j < t - 1; j++) r[j] = j;
    r[t-1]++;
    for (int j = 0; j < t; j++) s0 += fast_choose(I1.empty0[r[j]], j+1);
    mask = next_subset(mask);
  }
};

// ---------------------------------------------------------------------------------------------
// Bookkeeping records

struct chunk_t { int first_page[2]; };  // Second stream used by merged mode only
struct tile_t {
  int first_chunk = 0, nchunks = 0;  // Into the slice's chunk array
  int slot = -1;                     // Decoded staging slot, or -1
  int consumers = 0;                 // Remaining output tiles reading this tile
};
struct slot_t { int tile = -1; int64_t last_used = 0; };  // Staging slot

// Per-slice tables indexed by absolute set0 index
struct s0_tables_t {
  int n = -1, count = 0, nchild = 0;   // nchild = spots - k0
  halfsuper_s* wins0 = nullptr;        // Immediate wins of the mover's side
  uint32_t* child = nullptr;           // [count][nchild]: row of the child after placing at relative empty i
  uint8_t* rel = nullptr;              // [count][spots]: relative index of an absolute spot among non-set0 spots
};
// set1_info_t for all rows with a given row pattern
struct row_block_t {
  int n = -1, pattern = -1, rowbase = 0;
  int64_t last_used = 0;
  set1_info_t* infos = nullptr;
};

// ---------------------------------------------------------------------------------------------
// The engine

struct engine_t {
  const tiled_options_t& opts;
  const high_board_t board;
  tiled_stats_t stats;
  const int S, k, ntiles, low;
  info_t I;

  // Sizes
  int max_rows = 0, max_chunks = 0, max_tile_chunks = 0, nslots = 0;
  int64_t max_tile_entries = 0, max_s0_bytes = 0, max_wins0_bytes = 0;
  static const int row_slots = 16;

  arena_t arena;
  pages_t pages;

  // Per slice tables (sets1p, wins1, cs1ps) for the two live slices only, by slice parity
  struct slice_tables_t {
    int n = -1;
    set_t* sets1p = nullptr;
    wins1_t* wins1 = nullptr;
    uint16_t* cs1ps = nullptr;
  };
  slice_tables_t slice_tables[2];
  uint8_t* slice_table_memory[2];
  int64_t max_slice_table_bytes = 0;

  struct slice_job_t { engine_t* E; slice_tables_t* t; int n; };
  static void slice_item(void* ctx, const int c, int) {
    auto& J = *static_cast<slice_job_t*>(ctx);
    const helper_t<> H{J.E->I, J.n};
    const int nwins = H.sets1().size, ncs = H.cs1ps_size();
    for (int i = c; i < nwins; i += 64) J.t->wins1[i] = mid_wins1(J.E->I, J.n, i);
    for (int i = c; i < ncs; i += 64) J.t->cs1ps[i] = make_cs1ps(J.E->I, J.t->sets1p, J.n, i);
  }
  const slice_tables_t& slice_table(const int n) {
    auto& t = slice_tables[n & 1];
    if (t.n == n) return t;
    const helper_t<> H{I, n};
    uint8_t* m = slice_table_memory[n & 1];
    t.n = n;
    t.sets1p = reinterpret_cast<set_t*>(m); m += (int64_t(H.sets1p().size) * sizeof(set_t) + 63) & ~63;
    t.wins1 = reinterpret_cast<wins1_t*>(m); m += (int64_t(H.sets1().size) * sizeof(wins1_t) + 63) & ~63;
    t.cs1ps = reinterpret_cast<uint16_t*>(m);
    const auto sets1p_ = H.sets1p();
    for (const int i : range(sets1p_.size)) t.sets1p[i] = get(sets1p_, i);  // make_cs1ps below reads these
    slice_job_t J{this, &t, n};
    pool.run(64, slice_item, &J);
    return t;
  }

  // Caches of derived tables.  s0_tables[0] is the full table of the output slice; s0_tables[1] holds
  // only the win masks of the input slice, copied out when that slice stopped being the output.
  s0_tables_t s0_tables[2];
  uint8_t* s0_memory[2];
  row_block_t row_blocks[row_slots];
  int64_t clock = 0;
  int64_t batch_clock = 0;  // Clock at the start of the current batch: anything used since is pinned

  // Compressed slices n+1 (in) and n (out)
  geom_t* geom[2];
  tile_t* tiles[2];
  chunk_t* chunks[2];
  int nchunks_used[2] = {0, 0};
  int in = 0, out = 1;

  // Dense staging: halfsuper_s entries, or in merged mode {Vw, Vn} pairs where a parent ORs the
  // child's Vn into its win set and the child's Vw into its not-lose set
  uint8_t* staging;
  slot_t* slots;

  // Per work item scratch
  uint8_t* caches;  // One hs_caches_t per pool participant
  int* order;
  hs_caches_t* cache(const int c) const { return reinterpret_cast<hs_caches_t*>(caches + size_t(c) * hs_caches_bytes()); }

  bool aggressive = false;
  halfsuper_s* results = nullptr;          // For n <= 1, two-pass mode
  halfsupers_t* pair_results = nullptr;    // For n <= 1, merged mode
  const bool merged;
  const int esize;                         // Bytes per staging entry: 16, or 32 in merged mode

  // Prefix spots: enough that the largest tile stays a few MB, but no more, since tiny tiles parallelize poorly
  static int pick_prefix(const tiled_options_t& opts, const int S) {
    return min(S, opts.prefix ? opts.prefix : max(3, min(6, S - 14)));
  }

  engine_t(const high_board_t board, const tiled_options_t& opts)
    : opts(opts), board(board), S(36 - board.count()), k(pick_prefix(opts, S)), ntiles(1 << 2*k), low(S - k), I(make_info(board)),
      merged(opts.merged), esize(opts.merged ? 32 : 16) {
    // Sizes of everything
    for (int n = 0; n <= S; n++) {
      const helper_t<> H{I, n};
      int nchunks = 0;
      for (int prow = 0; prow < 1 << k; prow++)
        for (int pcol = 0; pcol < 1 << k; pcol++) {
          const geom_t g = make_geom(S, k, n, prow, pcol);
          if (!g.valid) continue;
          max_tile_entries = max(max_tile_entries, g.entries());
          max_rows = max(max_rows, g.nrows);
          const int c = chunk_count(g);
          nchunks += c;
          max_tile_chunks = max(max_tile_chunks, c);
        }
      max_chunks = max(max_chunks, nchunks);
      max_s0_bytes = max(max_s0_bytes, int64_t(H.sets0().size) * (16 + 4 * (S - H.k0()) + S) + 3 * 64);
      max_wins0_bytes = max(max_wins0_bytes, int64_t(H.sets0().size) * 16 + 64);
    }
    nslots = k + 1 + 4;
    for (int n = 0; n <= S; n++) {
      const helper_t<> H{I, n};
      max_slice_table_bytes = max(max_slice_table_bytes, int64_t(sizeof(set_t)) * H.sets1p().size + int64_t(sizeof(wins1_t)) * H.sets1().size
                                                         + int64_t(sizeof(uint16_t)) * H.cs1ps_size() + 3 * 64);
    }
    // Reserve the arena
    const int ncaches = (merged ? 2 : 1) * max(1, opts.threads);  // Merged mode codes two streams at once
    const size_t table_bytes = 2 * max_slice_table_bytes;
    size_t bytes = table_bytes + max_s0_bytes + max_wins0_bytes + size_t(row_slots) * max_rows * sizeof(set1_info_t)
                 + size_t(nslots) * max_tile_entries * esize + 2 * (ntiles * (sizeof(geom_t) + sizeof(tile_t)) + max_chunks * sizeof(chunk_t))
                 + nslots * sizeof(slot_t)
                 + size_t(ncaches) * hs_caches_bytes() + ntiles * sizeof(int) + 64 * 32;
    arena.reserve(bytes);
    stats.arena_bytes = bytes;
#ifndef __wasm__
    if (opts.verbose)
      slog("arena %.1f MB: slice tables %.1f, mover tables %.1f, row blocks %.1f, staging %.1f, codec caches %.1f, bookkeeping %.1f (prefix %d, %d slots, max tile %lld entries)",
           bytes / double(1 << 20), table_bytes / double(1 << 20), (max_s0_bytes + max_wins0_bytes) / double(1 << 20),
           double(row_slots) * max_rows * sizeof(set1_info_t) / double(1 << 20), double(nslots) * max_tile_entries * esize / double(1 << 20),
           double(ncaches) * hs_caches_bytes() / double(1 << 20),
           (2.0 * (ntiles * (sizeof(geom_t) + sizeof(tile_t)) + max_chunks * sizeof(chunk_t)) + nslots * sizeof(slot_t) + ntiles * sizeof(int)) / double(1 << 20),
           k, nslots, (long long)max_tile_entries);
#endif
    slice_table_memory[0] = arena.alloc<uint8_t>(max_slice_table_bytes);
    slice_table_memory[1] = arena.alloc<uint8_t>(max_slice_table_bytes);
    s0_memory[0] = arena.alloc<uint8_t>(max_s0_bytes);
    s0_memory[1] = arena.alloc<uint8_t>(max_wins0_bytes);
    for (auto& r : row_blocks) r.infos = arena.alloc<set1_info_t>(max_rows);
    staging = arena.alloc<uint8_t>(size_t(nslots) * max_tile_entries * esize);
    slots = arena.alloc<slot_t>(nslots);
    for (int i = 0; i < 2; i++) {
      geom[i] = arena.alloc<geom_t>(ntiles);
      tiles[i] = arena.alloc<tile_t>(ntiles);
      chunks[i] = arena.alloc<chunk_t>(max_chunks);
    }
    caches = arena.alloc<uint8_t>(size_t(ncaches) * hs_caches_bytes());
    order = arena.alloc<int>(ntiles);
  }
  ~engine_t() {
    pages.destroy();
    arena.release();
  }

  int64_t used_bytes() const { return int64_t(arena.used) + int64_t(pages.used()) * pages_t::page_words * 4; }
  void account() {
    stats.peak_bytes = max(stats.peak_bytes, used_bytes());
    stats.peak_pages = max(stats.peak_pages, int64_t(pages.used()) * pages_t::page_words * 4);
  }

  int tile_id(const int prow, const int pcol) const { return prow << k | pcol; }

  // ---- derived tables

  struct s0_job_t { engine_t* E; s0_tables_t* t; int n; };
  static void s0_item(void* ctx, const int c, int) {
    auto& J = *static_cast<s0_job_t*>(ctx);
    auto& t = *J.t;
    const int S = J.E->S;
    const engine_t& E = *J.E;
    const helper_t<> H{E.I, J.n};
    const auto sets0 = H.sets0();
    const int k0 = H.k0();
    for (int s0 = c; s0 < t.count; s0 += 64) {
      // As in make_set0_info (internal.h), without the offset0 table this engine never reads
      const set_t set0 = get(sets0, s0);
      const side_t side0 = H.root0() | side(E.I.empty, sets0, set0);
      t.wins0[s0] = halfsuper_wins(side0, H.parity()).s;
      uint8_t* rel = &t.rel[int64_t(s0) * S];
      uint32_t* child = &t.child[int64_t(s0) * t.nchild];
      for (int e = 0; e < S; e++) rel[e] = 255;
      const auto free = side_mask & ~side0;
      int next = 0;
      for (int e = 0; e < S; e++)
        if (free & side_t(1) << E.I.empty.empty[e]) {
          rel[e] = next;
          // Row of the child with a stone added at e: e's rank among the k0+1 elements is j
          const int j = e - next;
          uint32_t row = choose(e, j+1);
          for (int a = 0; a < k0; a++) row += choose(set0 >> 5*a & 0x1f, a + (a >= j) + 1);
          child[next++] = row;
        }
    }
  }
  // Full table for the output slice n; the previous output's win masks move to the small region first
  const s0_tables_t& s0_table(const int n) {
    auto& t = s0_tables[0];
    if (t.n == n) return t;
    if (t.n == n + 1) {
      auto& w = s0_tables[1];
      w.n = t.n; w.count = t.count; w.nchild = 0; w.child = nullptr; w.rel = nullptr;
      w.wins0 = reinterpret_cast<halfsuper_s*>(s0_memory[1]);
      memcpy(w.wins0, t.wins0, size_t(t.count) * 16);
    }
    const helper_t<> H{I, n};
    t.n = n; t.count = H.sets0().size; t.nchild = S - H.k0();
    uint8_t* m = s0_memory[0];
    t.wins0 = reinterpret_cast<halfsuper_s*>(m); m += (int64_t(t.count) * 16 + 63) & ~63;
    t.child = reinterpret_cast<uint32_t*>(m); m += (int64_t(t.count) * t.nchild * 4 + 63) & ~63;
    t.rel = m;
    s0_job_t J{this, &t, n};
    pool.run(min(t.count, 64), s0_item, &J);
    return t;
  }
  // Win masks of the input slice m = n+1 (decoding needs nothing else per column)
  const s0_tables_t& wins0_table(const int m) {
    if (s0_tables[0].n == m) return s0_tables[0];
    GEODE_ASSERT(s0_tables[1].n == m);
    return s0_tables[1];
  }

  // Same as make_set1_info (internal.h), but offset1p via suffix sums: O(k0 (k1 + spots)) rather than
  // O(k0 spots k1) choose lookups
  set1_info_t make_row_info(const int n, const int s1) const {
    set1_info_t I1;
    const helper_t<> H{I, n};
    const auto sets1 = H.sets1();
    const int k0 = H.k0(), k1 = H.k1(), parity = H.parity();
    I1.s1 = s1;
    const set_t set1 = get(sets1, s1);
    const side_t side1 = H.root1() | side(I.empty, sets1, set1);
    I1.wins1.after = halfsuper_wins(side1, parity);
    I1.wins1.before = halfsuper_wins(side1, !parity);
    int next = 0;
    {
      const auto free = side_mask & ~side1;
      for (int i = 0; i < S; i++)
        if (free & side_t(1) << I.empty.empty[i])
          I1.empty0[next++] = i;
    }
    int v[MID_MAX_SPOTS];
    for (int i = 0; i < k1; i++) v[i] = set1 >> 5*i & 0x1f;
    for (int a = 0; a <= k0; a++) {
      uint16_t suffix[MID_MAX_SPOTS + 1];
      suffix[k1] = 0;
      for (int i = k1 - 1; i >= 0; i--)
        suffix[i] = suffix[i+1] + (v[i] > a ? fast_choose(v[i]-a-1, i) : 0);
      for (int q = 0; q < S - k1; q++)
        I1.offset1p[a * (S-k1) + q] = suffix[I1.empty0[q] - q];
    }
    return I1;
  }

  struct row_job_t { engine_t* E; row_block_t* r; int nrows; };
  static void row_item(void* ctx, const int c, int) {
    auto& J = *static_cast<row_job_t*>(ctx);
    for (int i = c; i < J.nrows; i += 64)
      J.r->infos[i] = J.E->make_row_info(J.r->n, J.r->rowbase + i);
  }
  const row_block_t& row_block(const int n, const int pattern, const int rowbase, const int nrows) {
    row_block_t* victim = &row_blocks[0];
    for (auto& r : row_blocks) {
      if (r.n == n && r.pattern == pattern) { r.last_used = ++clock; return r; }
      if (r.last_used < victim->last_used) victim = &r;
    }
    GEODE_ASSERT(victim->last_used <= batch_clock);  // Never evict a block the current batch still needs
    victim->n = n; victim->pattern = pattern; victim->rowbase = rowbase; victim->last_used = ++clock;
    row_job_t J{this, victim, nrows};
    pool.run(min(nrows, 64), row_item, &J);
    return *victim;
  }

  // ---- input tiles

  // Input tile ids needed by output tile (prow=P1, pcol=P0) of slice n, indexed by the relative spot
  // index i used by inner(): the i-th spot not in set0.  Since spots are sorted, the last k-|P0| of
  // these are the prefix spots not in P0, and the first low-q0 are the low spots not in set0.
  // Entries are -1 for spots that can never be empty (prefix spots in P1) or when no low spot is empty.
  void input_ids(const int n, const int P1, const int P0, int ids[MID_MAX_SPOTS]) const {
    const int k0 = n >> 1, q0 = k0 - __builtin_popcount(P0), q1 = (n - k0) - __builtin_popcount(P1);
    const int nlow = low - q0;  // low spots not in set0
    for (int i = 0; i < MID_MAX_SPOTS; i++) ids[i] = -1;
    if (nlow - q1 > 0)  // some low spot is empty
      for (int i = 0; i < nlow; i++) ids[i] = tile_id(P0, P1);
    int i = nlow;
    for (int p = 0; p < k; p++)
      if (!(P0 >> p & 1)) {
        if (!(P1 >> p & 1)) ids[i] = tile_id(P0 | 1 << p, P1);
        i++;
      }
  }

  halfsuper_s* slot_data(const int slot) const { return reinterpret_cast<halfsuper_s*>(staging + int64_t(slot) * max_tile_entries * esize); }

  // Staging slot for input tile id, decoding it if needed.  needed lists tiles that must not be evicted.
  struct decode_job_t {
    engine_t* E; int n, id; const geom_t* g; const tile_t* t; halfsuper_s* data;
    bool turn_aggressive; const s0_tables_t* S0; const row_block_t* R; sets_t sets0p; const wins1_t* wins1;
  };
  static void decode_item(void* ctx, const int c, const int me) {
    auto& J = *static_cast<decode_job_t*>(ctx);
    if (J.E->merged) { decode_item_merged(J, c, me); return; }
    const geom_t& g = *J.g;
    page_reader_t reader{&J.E->pages, J.E->chunks[J.E->in][J.t->first_chunk + c].first_page[0]};
    hs_caches_init(J.E->cache(me));
    hs_decoder_t dec(J.E->cache(me), page_reader_t::fetch, &reader);
    const int nchunks = J.t->nchunks;
    const int lo = chunk_lo(g, c, nchunks), hi = chunk_lo(g, c+1, nchunks);
    for (int s1 = lo; s1 < hi; s1++) {
      const wins1_t& w1 = J.wins1[s1];
      halfsuper_s* out = J.data + int64_t(s1 - g.rowbase) * g.ncols;
      if (J.turn_aggressive) {
        // Coded against the row-only mask ~wins1.after: nothing per column, and a run of ALL entries
        // is one constant value
        const halfsuper_t domain = ~w1.after;
        const halfsuper_s all = rmax(~domain) | w1.before;
        for (int col = 0; col < g.ncols;) {
          if (const int run = dec.take_run(g.ncols - col)) {
            for (int i = 0; i < run; i++) { out[col + i] = all; dec.push_history(to_bits(domain)); }
            dec.end_run(to_bits(domain));
            col += run;
          } else {
            const halfsuper_t us = from_bits(dec.get(to_bits(domain)));
            out[col++] = rmax(~us) | w1.before;
          }
        }
      } else {
        // Coded against the in-play mask: recover the mover's immediate wins per column
        const set1_info_t& I1 = J.R->infos[s1 - g.rowbase];
        // An ALL entry here is us = inplay | wins0, which still varies with the mover's fives per
        // column, so a run saves the codec call but not the per-column rmax
        cursor_t cur;
        cur.start(I1, J.sets0p.k, subset_mask(J.sets0p, g.colbase));
        for (int col = 0; col < g.ncols;) {
          if (const int run = dec.take_run(g.ncols - col)) {
            halfsuper_t inplay = {0};
            for (int i = 0; i < run; i++, cur.step(I1)) {
              const halfsuper_t wins0 = J.S0->wins0[cur.s0];
              inplay = ~(wins0 | w1.after);
              out[col + i] = rmax(w1.after & ~wins0) | w1.before;
              dec.push_history(to_bits(inplay));
            }
            dec.end_run(to_bits(inplay));
            col += run;
          } else {
            const halfsuper_t wins0 = J.S0->wins0[cur.s0];
            const halfsuper_t inplay = ~(wins0 | w1.after);
            const halfsuper_t us = from_bits(dec.get(to_bits(inplay))) | wins0;
            out[col++] = rmax(~us) | w1.before;
            cur.step(I1);
          }
        }
      }
    }
  }
  // Merged mode: the win set is coded against the row-only mask; when it is not ALL, the not-lose set
  // is coded as a subset of the tie candidates, the in-play rotations the mover does not win
  static void decode_item_merged(decode_job_t& J, const int c, const int me) {
    const geom_t& g = *J.g;
    const chunk_t& chunk = J.E->chunks[J.E->in][J.t->first_chunk + c];
    page_reader_t reader{&J.E->pages, chunk.first_page[0]}, treader{&J.E->pages, chunk.first_page[1]};
    hs_caches_init(J.E->cache(2*me));
    hs_caches_init(J.E->cache(2*me+1));
    hs_decoder_t dec(J.E->cache(2*me), page_reader_t::fetch, &reader);
    hs_decoder_t tdec(J.E->cache(2*me+1), page_reader_t::fetch, &treader);
    const int nchunks = J.t->nchunks;
    const int lo = chunk_lo(g, c, nchunks), hi = chunk_lo(g, c+1, nchunks);
    halfsupers_t* pairs = reinterpret_cast<halfsupers_t*>(J.data);
    for (int s1 = lo; s1 < hi; s1++) {
      const wins1_t& w1 = J.wins1[s1];
      const halfsuper_t domain = ~w1.after;
      halfsupers_t* out = pairs + int64_t(s1 - g.rowbase) * g.ncols;
      const set1_info_t& I1 = J.R->infos[s1 - g.rowbase];
      cursor_t cur;
      cur.start(I1, J.sets0p.k, subset_mask(J.sets0p, g.colbase));
      for (int col = 0; col < g.ncols;) {
        if (const int run = dec.take_run(g.ncols - col)) {
          // Win everywhere in play, so not-lose is forced: inplay | wins0
          for (int i = 0; i < run; i++, cur.step(I1)) {
            const halfsuper_t wins0 = J.S0->wins0[cur.s0];
            out[col + i].win = rmax(w1.after) | w1.before;
            out[col + i].notlose = rmax(w1.after & ~wins0) | w1.before;
            dec.push_history(to_bits(domain));
          }
          dec.end_run(to_bits(domain));
          col += run;
        } else {
          const halfsuper_t wins0 = J.S0->wins0[cur.s0];
          const halfsuper_t inplay = ~(wins0 | w1.after);
          const halfsuper_t win = from_bits(dec.get(to_bits(domain)));
          // The encoder skips the tie stream whenever win is ALL, however the win itself was coded
          // (it may have come through a far copy rather than an ALL run)
          const halfsuper_t cand = inplay & ~win;
          const halfsuper_t notlose = win == domain ? win | wins0 : win | from_bits(tdec.get(to_bits(cand))) | wins0;
          out[col].win = rmax(~win) | w1.before;
          out[col].notlose = rmax(~notlose) | w1.before;
          col++;
          cur.step(I1);
        }
      }
    }
  }

  struct compute_job_t {
    engine_t* E; int n, P1, P0; const geom_t* g; tile_t* t; io_t in[MID_MAX_SPOTS]; int ids[MID_MAX_SPOTS];
    bool turn_aggressive, done; const s0_tables_t* S0; const row_block_t* R; sets_t sets0p;
    const wins1_t* wins1; const uint16_t* cs1ps; int k0, k1; uint32_t all_spots; bool early_exit;
  };
  static void compute_item(void* ctx, const int c, const int me) {
    auto& J = *static_cast<compute_job_t*>(ctx);
    engine_t& E = *J.E;
    if (E.merged) { compute_item_merged(J, c, me); return; }
    const geom_t& g = *J.g;
    const int S = E.S, n = J.n;
    page_writer_t writer{&E.pages};
    writer.start();
    E.chunks[E.out][J.t->first_chunk + c].first_page[0] = writer.first;
    hs_caches_init(E.cache(me));
    hs_encoder_t enc(E.cache(me), page_writer_t::emit, &writer);
    const int nchunks = J.t->nchunks;
    const int lo = chunk_lo(g, c, nchunks), hi = chunk_lo(g, c+1, nchunks);
    for (int s1 = lo; s1 < hi; s1++) {
      const set1_info_t& I1 = J.R->infos[s1 - g.rowbase];
      const wins1_t& w1 = J.wins1[s1];
      // Absolute mask of set1's spots
      uint32_t set1 = J.all_spots;
      for (int j = 0; j < S - J.k1; j++) set1 &= ~(uint32_t(1) << I1.empty0[j]);
      // The plain loop beats the incremental cursor here: it is short, branchless, and unrolls
      uint32_t mask = subset_mask(J.sets0p, g.colbase);
      for (int col = 0; col < g.ncols; col++, mask = next_subset(mask)) {
        // Absolute set0 index and mask, and set1's index relative to set0, from the relative mask.
        // The offset table is 16 bit and relies on wraparound, so accumulate s1p in 16 bits too.
        uint32_t s0 = 0, set0 = 0;
        uint16_t s1p = uint16_t(I1.s1);
        for (uint32_t bits = mask, j = 0; bits; bits &= bits - 1, j++) {
          const int r = __builtin_ctz(bits), e = I1.empty0[r];
          s0 += fast_choose(e, j+1);
          set0 |= uint32_t(1) << e;
          s1p -= I1.offset1p[j * J.sets0p.n + r];
        }
        const uint32_t* child = &J.S0->child[int64_t(s0) * J.S0->nchild];
        const uint8_t* rel = &J.S0->rel[int64_t(s0) * S];
        const uint16_t* cs1p = &J.cs1ps[int64_t(s1p) * J.S0->nchild];
        // Consider each move in turn, as in inner().  Once every in-play rotation is covered the
        // remaining children cannot change the result, which happens early for most winning entries.
        const halfsuper_t wins0 = J.S0->wins0[s0];
        const halfsuper_t inplay = ~(wins0 | w1.after);
        halfsuper_t us = {0};
        if (J.done && !J.turn_aggressive)
          us = ~halfsuper_t(0);
        if (J.early_exit) {
          for (uint32_t empties = J.all_spots & ~(set0 | set1); empties; empties &= empties - 1) {
            const int i = rel[__builtin_ctz(empties)];
            us |= J.in[i](child[i], cs1p[i]);
            if (!(inplay & ~us)) break;
          }
        } else {
          for (uint32_t empties = J.all_spots & ~(set0 | set1); empties; empties &= empties - 1) {
            const int i = rel[__builtin_ctz(empties)];
            us |= J.in[i](child[i], cs1p[i]);
          }
        }
        const halfsuper_t mk = J.turn_aggressive ? ~w1.after : ~halfsuper_t(0);
        us = (inplay & us) | (wins0 & mk);
        if (n <= 1)
          E.results[n + s1p] = us;
        // On aggressive turns us is a subset of ~wins1.after, a row-only mask, so code against it and
        // the decoder needs nothing per column.  On passive turns us includes the rotations where
        // both sides have five, so code against the in-play mask and let the decoder add wins0 back.
        const halfsuper_t domain = J.turn_aggressive ? ~w1.after : inplay;
        enc.put(to_bits(us & domain), to_bits(domain));
      }
    }
    enc.finish();
  }

  static void compute_item_merged(compute_job_t& J, const int c, const int me) {
    engine_t& E = *J.E;
    const geom_t& g = *J.g;
    const int S = E.S, n = J.n;
    page_writer_t writer{&E.pages}, twriter{&E.pages};
    writer.start(); twriter.start();
    chunk_t& chunk = E.chunks[E.out][J.t->first_chunk + c];
    chunk.first_page[0] = writer.first;
    chunk.first_page[1] = twriter.first;
    hs_caches_init(E.cache(2*me));
    hs_caches_init(E.cache(2*me+1));
    hs_encoder_t enc(E.cache(2*me), page_writer_t::emit, &writer);
    hs_encoder_t tenc(E.cache(2*me+1), page_writer_t::emit, &twriter);
    const int nchunks = J.t->nchunks;
    const int lo = chunk_lo(g, c, nchunks), hi = chunk_lo(g, c+1, nchunks);
    for (int s1 = lo; s1 < hi; s1++) {
      const set1_info_t& I1 = J.R->infos[s1 - g.rowbase];
      const wins1_t& w1 = J.wins1[s1];
      uint32_t set1 = J.all_spots;
      for (int j = 0; j < S - J.k1; j++) set1 &= ~(uint32_t(1) << I1.empty0[j]);
      uint32_t mask = subset_mask(J.sets0p, g.colbase);
      for (int col = 0; col < g.ncols; col++, mask = next_subset(mask)) {
        uint32_t s0 = 0, set0 = 0;
        uint16_t s1p = uint16_t(I1.s1);
        for (uint32_t bits = mask, j = 0; bits; bits &= bits - 1, j++) {
          const int r = __builtin_ctz(bits), e = I1.empty0[r];
          s0 += fast_choose(e, j+1);
          set0 |= uint32_t(1) << e;
          s1p -= I1.offset1p[j * J.sets0p.n + r];
        }
        const uint32_t* child = &J.S0->child[int64_t(s0) * J.S0->nchild];
        const uint8_t* rel = &J.S0->rel[int64_t(s0) * S];
        const uint16_t* cs1p = &J.cs1ps[int64_t(s1p) * J.S0->nchild];
        const halfsuper_t wins0 = J.S0->wins0[s0];
        const halfsuper_t inplay = ~(wins0 | w1.after);
        // Both meanings at once: the mover wins if some child leaves the opponent unable to not-lose,
        // and does not lose if some child leaves the opponent unable to win
        halfsuper_t win = {0}, notlose = {0};
        if (J.done)
          notlose = ~halfsuper_t(0);
        for (uint32_t empties = J.all_spots & ~(set0 | set1); empties; empties &= empties - 1) {
          const int i = rel[__builtin_ctz(empties)];
          const halfsupers_t& ch = reinterpret_cast<const halfsupers_t*>(J.in[i].data)[int64_t(child[i]) * J.in[i].stride + cs1p[i]];
          win |= ch.notlose;
          notlose |= ch.win;
          if (J.early_exit && !(inplay & ~win) && !(inplay & ~notlose)) break;
        }
        win = (inplay & win) | (wins0 & ~w1.after);
        notlose = (inplay & notlose) | wins0;
        if (n <= 1)
          E.pair_results[n + s1p] = halfsupers_t{win, notlose};
        enc.put(to_bits(win), to_bits(~w1.after));
        if (win != ~w1.after) {
          const halfsuper_t cand = inplay & ~win;
          tenc.put(to_bits(notlose & cand), to_bits(cand));
        }
      }
    }
    enc.finish();
    tenc.finish();
  }

  // ---- batches: several output tiles of one row-pattern group, whose inputs all fit in staging

  static const int max_batch = 64;
  struct batch_t {
    int n = 0, P1 = 0;
    int out[max_batch], nout = 0;             // Output tile ids
    int in[MID_MAX_SPOTS + 1], nin = 0;       // Union of needed input tile ids
    decode_job_t djobs[MID_MAX_SPOTS + 1];    // One per input tile that needs decoding
    int dtile[MID_MAX_SPOTS + 1], nd = 0, doffsets[MID_MAX_SPOTS + 2];
    compute_job_t cjobs[max_batch];
    int coffsets[max_batch + 1];
  };
  batch_t batch;  // Large, so a member rather than a stack object

  static void decode_batch_item(void* ctx, const int i, const int me) {
    auto& B = *static_cast<batch_t*>(ctx);
    int t = 0;
    while (i >= B.doffsets[t+1]) t++;
    decode_item(&B.djobs[t], i - B.doffsets[t], me);
  }
  static void compute_batch_item(void* ctx, const int i, const int me) {
    auto& B = *static_cast<batch_t*>(ctx);
    int t = 0;
    while (i >= B.coffsets[t+1]) t++;
    compute_item(&B.cjobs[t], i - B.coffsets[t], me);
  }

  // Would adding output tile id to the batch keep its inputs within the staging slots?
  bool fits(const int n, const int P1, const int id) {
    int ids[MID_MAX_SPOTS];
    input_ids(n, P1, id & ((1 << k) - 1), ids);
    int extra = 0;
    for (int i = 0; i < S; i++)
      if (ids[i] >= 0 && std::find(batch.in, batch.in + batch.nin, ids[i]) == batch.in + batch.nin
          && std::find(ids, ids + i, ids[i]) == ids + i)
        extra++;
    return batch.nout < max_batch && batch.nin + extra <= nslots;
  }
  void add(const int n, const int P1, const int id) {
    int ids[MID_MAX_SPOTS];
    input_ids(n, P1, id & ((1 << k) - 1), ids);
    for (int i = 0; i < S; i++)
      if (ids[i] >= 0 && std::find(batch.in, batch.in + batch.nin, ids[i]) == batch.in + batch.nin)
        batch.in[batch.nin++] = ids[i];
    batch.out[batch.nout++] = id;
  }

  // Decode the batch's inputs and compute its outputs
  void run_batch() {
    batch_t& B = batch;
    const int n = B.n, P1 = B.P1;
    const int m = n + 1;
    batch_clock = clock;
    s0_table(n);  // Build the output slice's mover table first, moving the input slice's win masks aside
    // Assign staging slots to every needed input, evicting the least recently used unneeded tiles
    if (n < S) {
      for (int i = 0; i < B.nin; i++)
        GEODE_ASSERT(tiles[in][B.in[i]].consumers > 0 && geom[in][B.in[i]].valid);
      B.nd = 0;
      for (int i = 0; i < B.nin; i++) {
        tile_t& t = tiles[in][B.in[i]];
        if (t.slot >= 0) { slots[t.slot].last_used = ++clock; continue; }
        int slot = -1;
        for (int s_ = 0; s_ < nslots; s_++)
          if (slots[s_].tile < 0) { slot = s_; break; }
        if (slot < 0) {
          for (int s_ = 0; s_ < nslots; s_++) {
            bool pinned = false;
            for (int j = 0; j < B.nin; j++) pinned |= B.in[j] == slots[s_].tile;
            if (!pinned && (slot < 0 || slots[s_].last_used < slots[slot].last_used)) slot = s_;
          }
          GEODE_ASSERT(slot >= 0);
          tiles[in][slots[slot].tile].slot = -1;
        }
        slots[slot] = slot_t{B.in[i], ++clock};
        t.slot = slot;
        B.dtile[B.nd++] = B.in[i];
      }
      // Decode jobs for the tiles not already in staging
      const helper_t<> H{I, m};
      const bool turn_aggressive = aggressive ^ (m & 1);
      B.doffsets[0] = 0;
      for (int d = 0; d < B.nd; d++) {
        const int id = B.dtile[d];
        const geom_t& g = geom[in][id];
        const tile_t& t = tiles[in][id];
        // Merged mode needs the mover's tables on every turn
        const bool per_column = merged || !turn_aggressive;
        B.djobs[d] = decode_job_t{this, n, id, &g, &t, slot_data(t.slot), turn_aggressive,
                                  per_column ? &wins0_table(m) : nullptr,
                                  per_column ? &row_block(m, id >> k, g.rowbase, g.nrows) : nullptr,
                                  make_sets(S - H.k1(), H.k0()), slice_table(m).wins1};
        B.doffsets[d+1] = B.doffsets[d] + t.nchunks;
      }
#ifndef __wasm__
      const auto t0 = std::chrono::steady_clock::now();
#endif
      pool.run(B.doffsets[B.nd], decode_batch_item, &B);
#ifndef __wasm__
      stats.decode_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#endif
    }
    // Compute jobs
    const helper_t<> H{I, n};
    B.coffsets[0] = 0;
    for (int o = 0; o < B.nout; o++) {
      const int id = B.out[o];
      const int P0 = id & ((1 << k) - 1);
      const geom_t& g = geom[out][id];
      compute_job_t& J = B.cjobs[o];
      J.E = this; J.n = n; J.P1 = P1; J.P0 = P0; J.g = &g; J.t = &tiles[out][id];
      input_ids(n, P1, P0, J.ids);
      for (int i = 0; i < S; i++)
        if (J.ids[i] >= 0) {
          const geom_t& ig = geom[in][J.ids[i]];
          // In merged mode entries are pairs, so offsets are in units of two halfsuper_s
          J.in[i] = io_t{slot_data(tiles[in][J.ids[i]].slot) - (merged ? 2 : 1) * (int64_t(ig.rowbase) * ig.ncols + ig.colbase), ig.ncols};
        }
      J.turn_aggressive = aggressive ^ (n & 1);
      J.done = n == S;
      J.S0 = &s0_table(n);
      J.R = &row_block(n, P1, g.rowbase, g.nrows);
      J.k0 = H.k0(); J.k1 = H.k1();
      J.sets0p = make_sets(S - J.k1, J.k0);
      J.wins1 = slice_table(n).wins1;
      J.cs1ps = slice_table(n).cs1ps;
      J.all_spots = (uint32_t(1) << S) - 1;
      J.early_exit = opts.early_exit;
      B.coffsets[o+1] = B.coffsets[o] + J.t->nchunks;
      stats.entries += g.entries();
    }
#ifndef __wasm__
    const auto t1 = std::chrono::steady_clock::now();
#endif
    pool.run(B.coffsets[B.nout], compute_batch_item, &B);
#ifndef __wasm__
    stats.compute_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
#endif
    // Release input tiles whose consumers are all done
    for (int o = 0; o < B.nout; o++) {
      const compute_job_t& J = B.cjobs[o];
      for (int i = 0; i < S; i++)
        if (J.ids[i] >= 0 && std::find(J.ids, J.ids + i, J.ids[i]) == J.ids + i) {
          tile_t& it = tiles[in][J.ids[i]];
          if (--it.consumers == 0) {
            for (int c = 0; c < it.nchunks; c++)
              for (int str = 0; str < (merged ? 2 : 1); str++) pages.release(chunks[in][it.first_chunk + c].first_page[str]);
            if (it.slot >= 0) { slots[it.slot].tile = -1; it.slot = -1; }
          }
        }
    }
    account();
    B.nout = B.nin = B.nd = 0;
  }

  // Solve one pass (aggressive or not, or both in merged mode), filling results for n <= 1
  void pass(const bool aggressive_, halfsuper_s* results_, halfsupers_t* pair_results_ = nullptr) {
    aggressive = aggressive_;
    results = results_;
    pair_results = pair_results_;
    for (int i = 0; i < 2; i++) {
      s0_tables[i].n = -1;
      slice_tables[i].n = -1;
      for (int id = 0; id < ntiles; id++) { geom[i][id] = geom_t(); tiles[i][id] = tile_t(); }
    }
    for (auto& r : row_blocks) r.n = -1;
    for (int s = 0; s < nslots; s++) slots[s] = slot_t();
    in = 0; out = 1;
    for (int n = S; n >= 0; n--) {
#ifndef __wasm__
      const auto t0 = std::chrono::steady_clock::now();
#endif
      // Geometry, chunk assignment, and consumer counts
      int nchunks = 0;
      int64_t entries = 0;
      for (int id = 0; id < ntiles; id++) {
        geom[out][id] = make_geom(S, k, n, id >> k, id & ((1 << k) - 1));
        tiles[out][id] = tile_t();
        if (geom[out][id].valid) {
          tiles[out][id].first_chunk = nchunks;
          tiles[out][id].nchunks = chunk_count(geom[out][id]);
          nchunks += tiles[out][id].nchunks;
          entries += geom[out][id].entries();
        }
      }
      nchunks_used[out] = nchunks;
      (void)entries;  // Only reported by the verbose native build
      if (n < S) {
        for (int id = 0; id < ntiles; id++) tiles[in][id].consumers = 0;
        for (int id = 0; id < ntiles; id++)
          if (geom[out][id].valid) {
            int ids[MID_MAX_SPOTS];
            input_ids(n, id >> k, id & ((1 << k) - 1), ids);
            for (int i = 0; i < S; i++)
              if (ids[i] >= 0 && std::find(ids, ids + i, ids[i]) == ids + i)
                tiles[in][ids[i]].consumers++;
          }
        for (int id = 0; id < ntiles; id++)
          GEODE_ASSERT((tiles[in][id].nchunks > 0) == (tiles[in][id].consumers > 0));
      }
      // Pages grow on demand inside the phases; just make sure every chunk can start
      pages.ensure(nchunks + 1);
      // Group output tiles by row pattern: an input tile (A,B) is consumed exactly by the output tiles
      // with row pattern B, so this decodes each input tile once and frees it at the end of its group,
      // keeping roughly one slice live.  Within a group, go from most to fewest column stones so that
      // consecutive tiles share inputs.
      const int pmask = (1 << k) - 1;
      int norder = 0;
      for (int prow = 0; prow <= pmask; prow++)
        for (int stones = k; stones >= 0; stones--)
          for (int pcol = 0; pcol <= pmask; pcol++)
            if (__builtin_popcount(pcol) == stones && geom[out][tile_id(prow, pcol)].valid)
              order[norder++] = tile_id(prow, pcol);
      // Batch consecutive tiles of a group while their inputs fit in staging
      batch.nout = batch.nin = 0;
      for (int i = 0; i < norder; i++) {
        const int id = order[i], P1 = id >> k;
        if (batch.nout && (P1 != batch.P1 || !fits(n, P1, id)))
          run_batch();
        if (!batch.nout) { batch.n = n; batch.P1 = P1; }
        add(n, P1, id);
      }
      if (batch.nout)
        run_batch();
      for (int id = 0; id < ntiles; id++)
        GEODE_ASSERT(tiles[in][id].consumers == 0 && tiles[in][id].slot < 0);
#ifndef __wasm__
      if (opts.verbose)
        slog("pass %d slice %d: %lld entries, %.1f MB compressed live (%.2f bits/entry this slice), peak %.1f MB, %.2f s", aggressive, n,
             (long long)entries, pages.used() * pages_t::page_words * 4 / double(1 << 20),
             pages.used() * pages_t::page_words * 32.0 / max(int64_t(1), entries),
             stats.peak_bytes / double(1 << 20), std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
#endif
      std::swap(in, out);
    }
    // Free the last slice
    for (int id = 0; id < ntiles; id++) {
      tile_t& t = tiles[in][id];
      for (int c = 0; c < t.nchunks; c++)
        for (int str = 0; str < (merged ? 2 : 1); str++) pages.release(chunks[in][t.first_chunk + c].first_page[str]);
      t = tile_t();
    }
    GEODE_ASSERT(pages.used() == 0);
  }
};

#ifndef __wasm__
struct native_workers_t {
  std::thread* threads = nullptr;
  int count = 0;
  native_workers_t(const int total) {
    pool.threads = total;
    astore(&pool.stop, 0);
    astore(&pool.nworkers, 0);
    count = total - 1;
    if (count > 0) {
      threads = new std::thread[count];
      for (int i = 0; i < count; i++) threads[i] = std::thread([]() { pool.worker(); });
    }
  }
  ~native_workers_t() {
    astore(&pool.stop, 1);
    aadd(&pool.generation, 1);
    for (int i = 0; i < count; i++) threads[i].join();
    delete[] threads;
    pool.threads = 1;
  }
};
#endif

}  // namespace

Vector<halfsupers_t,1+MID_MAX_SPOTS> midsolve_tiled_internal(const high_board_t board, const tiled_options_t& options,
                                                            tiled_stats_t* stats) {
#ifndef __wasm__
  const auto t0 = std::chrono::steady_clock::now();
  native_workers_t workers(max(1, options.threads));
#else
  pool.threads = max(1, options.threads);
#endif
  GEODE_ASSERT(36 - board.count() <= MID_MAX_SPOTS && options.prefix >= 0 && options.prefix <= 8);
  engine_t E(board, options);
  Vector<halfsupers_t,1+MID_MAX_SPOTS> results;
  if (options.merged)
    E.pass(true, nullptr, results.data());
  else {
    Vector<halfsuper_s,1+MID_MAX_SPOTS> raw[2];
    for (const int aggressive : range(2))
      E.pass(aggressive, raw[aggressive].data());
    // Interleave results.  We need to swap win and notlose for 0 < i.
    for (const int i : range(results.size()))
      for (const int j : range(2))
        get(results[i], j) = raw[j ^ (i == 0)][i];
  }
#ifndef __wasm__
  E.stats.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#endif
  if (stats) *stats = E.stats;
  return results;
}

#ifndef __wasm__
mid_values_t midsolve_tiled(const high_board_t board, const tiled_options_t& options, tiled_stats_t* stats) {
  const auto supers = midsolve_tiled_internal(board, options, stats);
  mid_values_t results;
  midsolve_traverse(board, supers.data(), results);
  return results;
}
#else
// Entry points for JavaScript.  Workers call midsolve_tiled_worker once and never return; the main
// call passes the total number of participants including itself, and whether to use the faster but
// larger merged mode (about 1.5x the compressed memory).
WASM_EXPORT void midsolve_tiled_worker() {
  pool.worker();
}
WASM_EXPORT void midsolve_tiled(const raw_t raw, const int threads, const int merged, mid_values_t* results) {
  results->clear();
  const auto board = high_board_t::from_raw(raw);
  tiled_options_t options;
  options.threads = threads;
  options.merged = merged;
  const auto supers = midsolve_tiled_internal(board, options);
  midsolve_traverse(board, supers.data(), *results);
}
#endif

}  // namespace pentago
