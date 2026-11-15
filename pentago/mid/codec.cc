// Streaming codec for the midgame solver's per-position win sets

#include "pentago/mid/codec.h"
#include <algorithm>
namespace pentago {

// Statistics, accumulated with compiler atomics so the file stays freestanding
static uint64_t global_modes[8], global_bits, global_mtf_rank[64], global_xor_rank[64], global_xor_pop[32];
static inline void atomic_add(uint64_t* p, const uint64_t v) { if (v) __atomic_fetch_add(p, v, __ATOMIC_RELAXED); }
codec_counts_t codec_counts() {
  codec_counts_t c;
  for (int m = 0; m < 8; m++) c.modes[m] = global_modes[m];
  for (int r = 0; r < 64; r++) { c.mtf_rank[r] = global_mtf_rank[r]; c.xor_rank[r] = global_xor_rank[r]; }
  for (int r = 0; r < 32; r++) c.xor_pop[r] = global_xor_pop[r];
  c.bits = global_bits;
  return c;
}
void reset_codec_counts() {
  for (auto& m : global_modes) m = 0;
  for (auto& m : global_mtf_rank) m = 0;
  for (auto& m : global_xor_rank) m = 0;
  for (auto& m : global_xor_pop) m = 0;
  global_bits = 0;
}

namespace {

const int cache_size = 32, dcache_size = 16, max_points = 16;
const int history = 8192;              // Past values the decoder keeps; far references reach this far back
const int hash_bits = 14;              // Encoder-side candidate tables (positions by value hash)
// Prefix code lengths, ordered by typical frequency: the mode with n leading ones has n+1 bits (7 for raw)
const int mode_all = 0, mode_mtf = 1, mode_xor = 2, mode_prev = 3, mode_far = 4, mode_sparse = 5, mode_dmtf = 6, mode_raw = 7;
static_assert((cache_size & (cache_size - 1)) == 0 && (dcache_size & (dcache_size - 1)) == 0 && (history & (history - 1)) == 0);

inline bits128 operator^(const bits128 a, const bits128 b) { return bits128{a.lo ^ b.lo, a.hi ^ b.hi}; }
inline bits128 operator&(const bits128 a, const bits128 b) { return bits128{a.lo & b.lo, a.hi & b.hi}; }
inline bits128 operator|(const bits128 a, const bits128 b) { return bits128{a.lo | b.lo, a.hi | b.hi}; }
inline bits128 operator~(const bits128 a) { return bits128{~a.lo, ~a.hi}; }
inline bool is_zero(const bits128 a) { return !(a.lo | a.hi); }
inline int pop(const bits128 a) { return __builtin_popcountll(a.lo) + __builtin_popcountll(a.hi); }
inline bits128 bit(const int i) { return i < 64 ? bits128{uint64_t(1) << i, 0} : bits128{0, uint64_t(1) << (i - 64)}; }

// Sixteen byte lanes, via the GCC/clang generic vector extensions so the same code lowers to NEON, SSE, and wasm SIMD
typedef uint8_t u8x16 __attribute__((vector_size(16)));
inline u8x16 load16(const uint8_t* p) { u8x16 v; __builtin_memcpy(&v, p, 16); return v; }
inline void store16(uint8_t* p, const u8x16 v) { __builtin_memcpy(p, &v, 16); }
inline u8x16 splat16(const uint8_t x) { return u8x16{x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x}; }
inline u8x16 min16(const u8x16 a, const u8x16 b) { return a < b ? a : b; }

// Per-lane popcount.  Clang 19+ has __builtin_elementwise_popcount, but Ubuntu 24.04 ships clang 18 and GCC has
// no equivalent, so spell out the instruction where we know it: i8x16.popcnt on wasm, and on x86 two pshufb
// nibble lookups (what clang emits short of AVX512 BITALG).  Anything else falls back to bit twiddling.
inline u8x16 popcount16(const u8x16 v) {
#if PENTAGO_WASM_SIMD
  return (u8x16)wasm_i8x16_popcnt((v128_t)v);
#elif __has_builtin(__builtin_elementwise_popcount)
  return __builtin_elementwise_popcount(v);
#elif PENTAGO_SSE && defined(__SSSE3__)
  const __m128i nibbles = _mm_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  return (u8x16)_mm_shuffle_epi8(nibbles, (__m128i)(v & 15)) + (u8x16)_mm_shuffle_epi8(nibbles, (__m128i)(v >> 4));
#else
  const u8x16 a = v - ((v >> 1) & 0x55);
  const u8x16 b = (a & 0x33) + ((a >> 2) & 0x33);
  return (b + (b >> 4)) & 15;
#endif
}

// Smallest lane.  Clang has a builtin; GCC gets a tree of pairwise lane minimums
inline uint8_t min_lane(u8x16 v) {
#if __has_builtin(__builtin_reduce_min)
  return __builtin_reduce_min(v);
#else
  v = min16(v, __builtin_shufflevector(v, v, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15));
  v = min16(v, __builtin_shufflevector(v, v, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7));
  v = min16(v, __builtin_shufflevector(v, v, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3));
  v = min16(v, __builtin_shufflevector(v, v, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));
  return v[0];
#endif
}

}  // namespace

// Recency caches of 128 bit values in byte-plane layout: planes[k][j] is byte k of slot j.  One pass
// over the 16 planes then computes the masked popcount of every slot against a query at once, which
// serves both exact matching (popcount zero) and nearest neighbor search.  Ranks are coded flat, so
// slots are a ring: rank r is the r-th most recently inserted value.  Trivial, since hs_caches_init
// zeroes it, and outside the unnamed namespace since the externally visible hs_caches_t embeds it
// (GCC's -Wsubobject-linkage rejects fields of anonymous type there).
template<int size> struct plane_cache_t {
  static_assert(size == 16 || size == 32);
  uint8_t planes[16][size];
  int head;  // Next slot to fill; slot of rank r is (head-1-r) mod size

  int slot(const int rank) const { return (head - 1 - rank) & (size - 1); }
  int rank(const int slot) const { return (head - 1 - slot) & (size - 1); }

  bits128 get(const int rank) const {
    const int j = slot(rank);
    uint8_t b[16];
    for (int k = 0; k < 16; k++) b[k] = planes[k][j];
    bits128 v; __builtin_memcpy(&v, b, 16); return v;
  }
  void insert(const bits128 x) {
    uint8_t b[16]; __builtin_memcpy(b, &x, 16);
    for (int k = 0; k < 16; k++) planes[k][head] = b[k];
    head = (head + 1) & (size - 1);
  }

  // pops[j] = popcount((slot j ^ x) & ip) for every slot; returns the smallest popcount
  int scan(const bits128 x, const bits128 ip, uint8_t pops[size]) const {
    u8x16 xv, iv;
    __builtin_memcpy(&xv, &x, 16);
    __builtin_memcpy(&iv, &ip, 16);
    u8x16 s0 = splat16(0), s1 = splat16(0);
#define PENTAGO_SCAN_STEP(k) { \
      const u8x16 xk = __builtin_shufflevector(xv, xv, k, k, k, k, k, k, k, k, k, k, k, k, k, k, k, k); \
      const u8x16 ik = __builtin_shufflevector(iv, iv, k, k, k, k, k, k, k, k, k, k, k, k, k, k, k, k); \
      s0 += popcount16((load16(planes[k]) ^ xk) & ik); \
      if (size == 32) s1 += popcount16((load16(planes[k] + 16) ^ xk) & ik); }
    PENTAGO_SCAN_STEP(0) PENTAGO_SCAN_STEP(1) PENTAGO_SCAN_STEP(2) PENTAGO_SCAN_STEP(3)
    PENTAGO_SCAN_STEP(4) PENTAGO_SCAN_STEP(5) PENTAGO_SCAN_STEP(6) PENTAGO_SCAN_STEP(7)
    PENTAGO_SCAN_STEP(8) PENTAGO_SCAN_STEP(9) PENTAGO_SCAN_STEP(10) PENTAGO_SCAN_STEP(11)
    PENTAGO_SCAN_STEP(12) PENTAGO_SCAN_STEP(13) PENTAGO_SCAN_STEP(14) PENTAGO_SCAN_STEP(15)
#undef PENTAGO_SCAN_STEP
    store16(pops, s0);
    if (size == 32) store16(pops + 16, s1);
    return min_lane(size == 32 ? min16(s0, s1) : s0);
  }

  // Rank of the most recent slot with pops == 0, given that one exists
  int exact(const uint8_t pops[size]) const {
    int r = 0;
    while (pops[slot(r)]) r++;
    return r;
  }
  // Rank of the most recent slot with pops equal to the known minimum
  int nearest(const uint8_t pops[size], const int min_pop) const {
    int r = 0;
    while (pops[slot(r)] != min_pop) r++;
    return r;
  }
};

struct hs_caches_t {
  bits128 prev;
  plane_cache_t<cache_size> cache;
  plane_cache_t<dcache_size> dcache;
  // Every entry so far, most recent history of them addressable by distance
  uint64_t pos;
  bits128 hist[history];
  // Encoder only: last position + 1 (0 = none) of each whole value and of each 32 bit quarter
  uint32_t exact_hash[1 << hash_bits];
  uint32_t quarter_hash[4][1 << hash_bits];

  const bits128& back(const uint64_t distance) const { return hist[(pos - distance) & (history - 1)]; }
  void push(const bits128 x) { hist[pos & (history - 1)] = x; pos++; }
  static uint32_t hash_value(const bits128 x) { return uint32_t((x.lo * 0x9E3779B97F4A7C15ull ^ x.hi * 0xC2B2AE3D27D4EB4Full) >> (64 - hash_bits)); }
  static uint32_t hash_quarter(const uint32_t q) { return (q * 0x9E3779B9u) >> (32 - hash_bits); }
  uint32_t quarter(const bits128 x, const int q) const { return uint32_t((q < 2 ? x.lo : x.hi) >> (32 * (q & 1))); }
  void index(const bits128 x) {  // Record the value just pushed (at pos-1) in the encoder tables
    exact_hash[hash_value(x)] = uint32_t(pos);
    for (int q = 0; q < 4; q++) quarter_hash[q][hash_quarter(quarter(x, q))] = uint32_t(pos);
  }
};
int hs_caches_bytes() { return sizeof(hs_caches_t); }
void hs_caches_init(hs_caches_t* caches) { __builtin_memset(caches, 0, sizeof(hs_caches_t)); }  // All zeros is the initial state

hs_encoder_t::hs_encoder_t(hs_caches_t* caches, const emit_word_t emit, void* ctx) : state(caches), emit(emit), ctx(ctx) {}
hs_encoder_t::~hs_encoder_t() {
  for (int m = 0; m < 8; m++) atomic_add(&global_modes[m], counts.modes[m]);
  for (int r = 0; r < 64; r++) { atomic_add(&global_mtf_rank[r], counts.mtf_rank[r]); atomic_add(&global_xor_rank[r], counts.xor_rank[r]); }
  for (int r = 0; r < 32; r++) atomic_add(&global_xor_pop[r], counts.xor_pop[r]);
  atomic_add(&global_bits, counts.bits);
}
hs_decoder_t::hs_decoder_t(hs_caches_t* caches, const fetch_word_t fetch, void* ctx) : state(caches), fetch(fetch), ctx(ctx) {}
hs_decoder_t::~hs_decoder_t() {}

void hs_encoder_t::flush() {
  emit(ctx, uint32_t(acc));
  words++;
  acc >>= 32;
  nacc -= 32;
}

void hs_encoder_t::write(const uint64_t value, const int bits) {
  acc |= value << nacc;
  nacc += bits;
  if (nacc >= 32)
    flush();
}

void hs_decoder_t::refill() {
  while (nacc <= 32) {
    acc |= uint64_t(fetch(ctx)) << nacc;
    nacc += 32;
  }
}

uint64_t hs_decoder_t::read(const int bits) {
  if (nacc < bits) refill();
  const uint64_t value = acc & ((uint64_t(1) << bits) - 1);
  acc >>= bits;
  nacc -= bits;
  return value;
}

int hs_decoder_t::read_gamma() {
  int n = 0;
  while (!read(1)) n++;
  int v = 1;
  for (int b = 0; b < n; b++) v = v << 1 | int(read(1));
  return v;
}

int hs_decoder_t::take_run(const int limit) {
  if (!run_all) {
    if (run_far) return 0;  // Inside a far copy: entries come from get()
    if (nacc < 1) refill();
    if (acc & 1) return 0;  // Next mode is not ALL
    acc >>= 1; nacc--;
    run_all = read_gamma();
  }
  const int r = std::min(run_all, limit);
  run_all -= r;
  return r;
}

// The caller produced r ALL entries itself; keep the history and context in step
void hs_decoder_t::end_run(const bits128 last_inplay) {
  state->prev = last_inplay;
}
void hs_decoder_t::push_history(const bits128 inplay) {
  state->push(inplay);
}

int hs_decoder_t::read_unary() {
  if (nacc < 8) refill();
  const int ones = std::min(7, __builtin_ctzll(~acc));
  const int consumed = ones < 7 ? ones + 1 : 7;
  acc >>= consumed;
  nacc -= consumed;
  return ones;
}

// Elias gamma: floor(log2 v) zeros, then the bits of v from its top bit down.  Values are < 2^20.

// Elias gamma: floor(log2 v) zeros, then the bits of v from its top bit down.  Values are < 2^24.
static inline int gamma_length(const int v) { return 31 - __builtin_clz(v); }
void hs_encoder_t::write_gamma(const int v) {
  const int n = gamma_length(v);
  write(0, n);
  for (int b = n; b >= 0; b--) write(v >> b & 1, 1);
}
static inline int gamma_bits(const int v) { return 2 * gamma_length(v) + 1; }

// Runs of ALL entries and far exact copies are written as one token when they end
void hs_encoder_t::flush_pending() {
  if (pending_all) {
    write(0, 1);  // Mode ALL
    write_gamma(pending_all);
    counts.modes[mode_all] += pending_all;
    pending_all = 0;
  }
  if (pending_far) {
    write((uint64_t(1) << mode_far) - 1, mode_far + 1);
    write_gamma(far_distance);
    write(0, 1);  // Exact copy
    write_gamma(pending_far);
    counts.modes[mode_far] += pending_far;
    pending_far = 0;
  }
}

void hs_encoder_t::put(const bits128 x, const bits128 ip) {
  auto& S = *state;
  // Continue a pending run if possible
  if (pending_far && is_zero((S.back(far_distance) ^ x) & ip)) {
    pending_far++;
    S.prev = x; S.push(x); S.index(x);
    return;
  }
  if (is_zero(x ^ ip) && !pending_far) {
    pending_all++;
    S.prev = x; S.push(x); S.index(x);
    return;
  }
  flush_pending();
  if (is_zero(x ^ ip)) {
    pending_all = 1;
    S.prev = x; S.push(x); S.index(x);
    return;
  }
  // Mode prefix plus a fixed payload in one write
  const auto mode = [this](const int m, const uint64_t payload = 0, const int bits = 0) {
    const int len = m < 7 ? m + 1 : 7;
    write(((uint64_t(1) << m) - 1) | payload << len, len + bits);
    counts.modes[m]++;
  };
  // Up to 16 indices of 7 bits, packed four at a time
  const auto points = [this](bits128 p) {
    uint64_t packed = 0; int n = 0;
    for (int w = 0; w < 2; w++)
      for (uint64_t word = w ? p.hi : p.lo; word; word &= word - 1) {
        packed |= uint64_t(__builtin_ctzll(word) + 64 * w) << 7 * n;
        if (++n == 4) { write(packed, 28); packed = 0; n = 0; }
      }
    if (n) write(packed, 7 * n);
  };
  if (is_zero((x ^ S.prev) & ip)) {
    mode(mode_prev);
  } else {
    uint8_t pops[cache_size];
    const int minpop = S.cache.scan(x, ip, pops);
    const bits128 d = x ^ S.prev;
    if (minpop == 0) {
      const int hit = S.cache.exact(pops);
      mode(mode_mtf, hit, 5); counts.mtf_rank[hit]++;
    } else {
      // Far exact copy: the most recent entry with this value, if within reach and equal modulo the mask
      const uint32_t ex = S.exact_hash[S.hash_value(x)];
      const uint64_t exact_distance = ex ? S.pos - (ex - 1) : 0;
      if (exact_distance && exact_distance < history && is_zero((S.back(exact_distance) ^ x) & ip)) {
        // Start a run; written when it ends
        far_distance = int(exact_distance);
        pending_far = 1;
        S.prev = x; S.push(x); S.index(x);
        return;
      }
      uint8_t dpops[dcache_size];
      if (S.dcache.scan(d, ip, dpops) == 0) {
        const int dhit = S.dcache.exact(dpops);
        mode(mode_dmtf, dhit, 4);
      } else {
        // Literal: the cheapest of sparse, xor against the closest cache entry, xor against the closest
        // far candidate sharing a quarter, and raw
        const int p1 = pop(x), p2 = pop(ip & ~x);
        const int psparse = std::min(p1, p2);
        const int bpop = minpop, best = S.cache.nearest(pops, minpop);
        int far_pop = 1000, far_d = 0;
        for (int q = 0; q < 4; q++) {
          const uint32_t c = S.quarter_hash[q][S.hash_quarter(S.quarter(x, q))];
          if (!c) continue;
          const uint64_t dist = S.pos - (c - 1);
          if (dist >= history) continue;
          const int p = pop((S.back(dist) ^ x) & ip);
          if (p < far_pop || (p == far_pop && int(dist) < far_d)) { far_pop = p; far_d = int(dist); }
        }
        if (far_pop == 0) {
          // A quarter candidate that matches modulo the mask: a far exact copy, possibly starting a run
          far_distance = far_d;
          pending_far = 1;
          S.prev = x; S.push(x); S.index(x);
          return;
        }
        const int cost_sparse = psparse <= 31 ? 6 + 1 + 5 + 7 * psparse : 1000,
                  cost_xor = bpop <= max_points ? 3 + 5 + 4 + 7 * bpop : 1000,
                  cost_far = far_pop <= max_points ? 5 + gamma_bits(std::max(far_d, 1)) + 1 + 4 + 7 * far_pop : 1000,
                  cost_raw = 7 + 128;
        const int cost_best = std::min({cost_sparse, cost_xor, cost_far, cost_raw});
        if (cost_best == cost_sparse) {
          const bool pol = p2 < p1;
          mode(mode_sparse, uint64_t(pol) | uint64_t(psparse) << 1, 6);
          points(pol ? ip & ~x : x);
        } else if (cost_best == cost_xor) {
          mode(mode_xor, uint64_t(best) | uint64_t(bpop - 1) << 5, 9); counts.xor_rank[best]++; counts.xor_pop[bpop]++;
          points((S.cache.get(best) ^ x) & ip);
        } else if (cost_best == cost_far) {
          mode(mode_far);
          write_gamma(far_d);
          write(1, 1);  // Xor, not an exact copy
          write(far_pop - 1, 4);
          points((S.back(far_d) ^ x) & ip);
        } else {
          mode(mode_raw);
          for (int w = 0; w < 4; w++)
            write(uint32_t((w < 2 ? x.lo : x.hi) >> (32 * (w & 1))), 32);
        }
        S.cache.insert(x);
        S.dcache.insert(d);
      }
    }
  }
  S.prev = x;
  S.push(x);
  S.index(x);
}

uint64_t hs_encoder_t::finish() {
  flush_pending();
  const uint64_t bits = 32 * words + nacc;
  counts.bits += bits;
  while (nacc > 0)
    flush();
  return bits;
}

bits128 hs_decoder_t::get(const bits128 ip) {
  auto& S = *state;
  bits128 x;
  if (run_all) {
    run_all--;
    x = ip;
  } else if (run_far) {
    run_far--;
    x = S.back(far_distance) & ip;
  } else {
    const auto points = [this]() {
      const int count = read(4) + 1;
      bits128 p = {0, 0};
      // Up to 16 indices of 7 bits: pull them in batches of 4 (28 bits) to amortize refills
      for (int i = 0; i < count; i += 4) {
        const int m = std::min(4, count - i);
        uint64_t v = read(7 * m);
        for (int j = 0; j < m; j++, v >>= 7)
          p = p | bit(v & 127);
      }
      return p;
    };
    const int mode = read_unary();
    switch (mode) {
      case mode_all: run_all = read_gamma() - 1; x = ip; break;
      case mode_prev: x = S.prev & ip; break;
      case mode_mtf: x = S.cache.get(read(5)) & ip; break;
      case mode_dmtf: x = (S.prev ^ S.dcache.get(read(4))) & ip; break;
      case mode_far: {
        far_distance = read_gamma();
        if (!read(1)) {  // Exact copy, possibly a run
          run_far = read_gamma() - 1;
          x = S.back(far_distance) & ip;
        } else {
          const int count = read(4) + 1;
          bits128 p = {0, 0};
          for (int i = 0; i < count; i += 4) {
            const int m = std::min(4, count - i);
            uint64_t v = read(7 * m);
            for (int j = 0; j < m; j++, v >>= 7)
              p = p | bit(v & 127);
          }
          x = (S.back(far_distance) ^ p) & ip;
          S.cache.insert(x);
          S.dcache.insert(x ^ S.prev);
        }
        break;
      }
      default: {  // sparse, raw
        if (mode == mode_sparse) {
          const bool pol = read(1);
          const int count = read(5);
          bits128 p = {0, 0};
          for (int i = 0; i < count; i += 4) {
            const int m = std::min(4, count - i);
            uint64_t v = read(7 * m);
            for (int j = 0; j < m; j++, v >>= 7)
              p = p | bit(v & 127);
          }
          x = pol ? ip & ~p : p;
        } else if (mode == mode_xor) {
          const bits128 c = S.cache.get(read(5));
          x = (c ^ points()) & ip;
        } else {
          uint64_t w[4];
          for (int i = 0; i < 4; i++) w[i] = read(32);
          x = {w[0] | w[1] << 32, w[2] | w[3] << 32};
        }
        S.cache.insert(x);
        S.dcache.insert(x ^ S.prev);
        break;
      }
    }
  }
  S.prev = x;
  S.push(x);
  return x;
}

}  // namespace pentago
