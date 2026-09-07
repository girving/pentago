// Streaming codec for the midgame solver's per-position win sets
//
// Each entry of a midgame slice is a 128 bit halfsuper.  The stored value is rmax(~us) | wins1.before,
// which the decoder can rebuild from x = us & inplay given the immediate win masks wins0 and wins1
// (functions of the two stone sets alone).  We therefore code x relative to the known in-play mask,
// using a small set of token modes chosen by measurement on real slices:
//
//   0        ALL     x == inplay (the mover wins everywhere still in play), then a gamma coded run length
//   10       MTF     x == recent distinct value, 5 bit rank into a 32 entry recency cache
//   110      XOR     5 bit cache rank, 4 bit count c-1, then c 7 bit indices where x differs from the cache entry
//   1110     PREV    x == previous entry (within inplay)
//   11110    FAR     gamma coded distance into the last 8192 entries, then either an exact copy with a
//                    gamma coded run length, or 4 bit count c-1 and c indices where x differs from it
//   111110   SPARSE  1 bit polarity, 5 bit count c, then c 7 bit rotation indices of x or inplay & ~x
//   1111110  DMTF    x ^ prev == recent delta, 4 bit rank into a 16 entry cache
//   1111111  RAW     128 bits
//
// The decoder keeps a ring of the last 8192 values, so far references are just distances; only the
// encoder searches, using hash tables of whole values and of 32 bit quarters to find copies and near
// neighbors far beyond the recency cache.  Runs (of ALL, or of far copies) are single tokens.
//
// Equality is always tested modulo the current inplay mask, so a cached value from a different
// position matches whenever it agrees on the rotations that matter here.  Streams are independent
// per chunk, so chunks can be encoded and decoded in parallel.
#pragma once

#include "pentago/mid/halfsuper.h"
#include <cstdint>
namespace pentago {

// 128 bits as two words, so the codec does not depend on the halfsuper representation
struct bits128 {
  uint64_t lo, hi;
};
static inline bits128 to_bits(const halfsuper_s h) { bits128 b; __builtin_memcpy(&b, &h, 16); return b; }
static inline halfsuper_s from_bits(const bits128 b) { halfsuper_s h; __builtin_memcpy(&h, &b, 16); return h; }

// Global counts of encoder decisions, for tuning (atomically accumulated when an encoder finishes)
struct codec_counts_t {
  uint64_t modes[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // Entries per mode
  uint64_t bits = 0;                             // Total bits written
  uint64_t mtf_rank[64] = {0}, xor_rank[64] = {0}, xor_pop[32] = {0};  // Histograms
};
codec_counts_t codec_counts();  // TEMPORARY: last mode seen (-1 pending run, -2 continued run)
void reset_codec_counts();

// Output is produced and consumed as 32 bit words through callbacks, so streams can live in any
// storage (a growable buffer in tests, chained pages in the solver) without the codec allocating.
typedef void (*emit_word_t)(void* ctx, uint32_t word);
typedef uint32_t (*fetch_word_t)(void* ctx);

// Shared recency caches; opaque here, defined in codec.cc.  Large, so callers keep one per stream.
struct hs_caches_t;
int hs_caches_bytes();
void hs_caches_init(hs_caches_t* caches);

class hs_encoder_t {
public:
  // caches must be initialized with hs_caches_init and outlive the encoder
  hs_encoder_t(hs_caches_t* caches, emit_word_t emit, void* ctx);
  ~hs_encoder_t();

  // Append one entry.  x must be a subset of inplay.
  void put(const bits128 x, const bits128 inplay);

  // Flush pending bits (padding with zeros) and return the total number of bits written
  uint64_t finish();

private:
  hs_caches_t* state;
  emit_word_t emit;
  void* ctx;
  codec_counts_t counts;
  int pending_all = 0;  // Consecutive ALL entries not yet written as a run token
  int pending_far = 0;  // Entries of a far exact copy not yet written, at distance far_distance
  int far_distance = 0;
  uint64_t words = 0;  // Words emitted so far
  uint64_t acc = 0;    // Pending bits, least significant first
  int nacc = 0;

  void write(uint64_t value, int bits);  // bits <= 32
  void write_gamma(int v);
  void flush();
  void flush_pending();
};

class hs_decoder_t {
public:
  hs_decoder_t(hs_caches_t* caches, fetch_word_t fetch, void* ctx);
  ~hs_decoder_t();

  // Decode the next entry given its inplay mask
  bits128 get(const bits128 inplay);

  // If the next token is a run of ALL entries, consume up to limit of them and return how many, else
  // 0; runs longer than limit continue on later calls.  The caller then produces those entries itself
  // (each equals its inplay mask) and reports the last inplay mask via end_run so that the
  // previous-entry context stays in step.
  int take_run(int limit);
  void end_run(const bits128 last_inplay);
  // Record an entry the caller produced itself during an ALL run (its value is its inplay mask)
  void push_history(const bits128 inplay);

private:
  hs_caches_t* state;
  fetch_word_t fetch;
  void* ctx;
  uint64_t acc = 0;  // Pending bits, least significant first
  int nacc = 0;

  void refill();             // Ensure at least 32 pending bits
  uint64_t read(int bits);   // bits <= 32
  int read_unary();          // Number of leading one bits before a zero, capped at 7
  int read_gamma();          // Elias gamma code, values >= 1
  int run_all = 0;           // Remaining entries of an ALL run not yet handed out through get
  int run_far = 0;           // Remaining entries of a far exact copy
  int far_distance = 0;
};

}  // namespace pentago
