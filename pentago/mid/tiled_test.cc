// Tests for the tiled, compressed midgame solver

#include "pentago/mid/tiled.h"
#include "pentago/mid/codec.h"
#include "pentago/mid/midengine.h"
#include "pentago/base/board.h"
#include "pentago/utility/random.h"
#include "pentago/utility/log.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

// Random in-play masks and subsets with the kinds of structure the codec targets
TEST(codec, roundtrip) {
  Random random(7);
  for (const int trial : range(20)) {
    const int n = 1000 + trial;
    std::vector<bits128> xs(n), ips(n);
    bits128 prev = {0, 0};
    for (int i = 0; i < n; i++) {
      bits128 ip = {random.bits<uint64_t>() | random.bits<uint64_t>(), random.bits<uint64_t>() | random.bits<uint64_t>()};
      bits128 x;
      switch (random.uniform<int>(0, 8)) {
        case 0: x = ip; break;
        case 1: x = {0, 0}; break;
        case 2: x = {prev.lo & ip.lo, prev.hi & ip.hi}; break;
        case 3: x = {random.bits<uint64_t>() & ip.lo, random.bits<uint64_t>() & ip.hi}; break;
        case 4: x = {(prev.lo ^ (uint64_t(1) << random.uniform<int>(0, 64))) & ip.lo, prev.hi & ip.hi}; break;
        case 5: x = {ip.lo & ~(uint64_t(1) << random.uniform<int>(0, 64)), ip.hi}; break;
        case 6: x = {(uint64_t(1) << random.uniform<int>(0, 64)) & ip.lo, 0}; break;
        default: x = xs[random.uniform<int>(0, i + 1) % (i + 1)]; x = {x.lo & ip.lo, x.hi & ip.hi}; break;
      }
      xs[i] = x; ips[i] = ip; prev = x;
    }
    // Streams live in a vector of words here
    std::vector<uint32_t> words;
    const auto emit = [](void* ctx, const uint32_t w) { static_cast<std::vector<uint32_t>*>(ctx)->push_back(w); };
    struct reader_t { const std::vector<uint32_t>* words; size_t pos; };
    const auto fetch = [](void* ctx) { auto& r = *static_cast<reader_t*>(ctx); return r.pos < r.words->size() ? (*r.words)[r.pos++] : 0u; };
    std::vector<uint8_t> caches(hs_caches_bytes());
    hs_caches_init(reinterpret_cast<hs_caches_t*>(caches.data()));
    {
      hs_encoder_t enc(reinterpret_cast<hs_caches_t*>(caches.data()), emit, &words);
      for (int i = 0; i < n; i++) enc.put(xs[i], ips[i]);
      enc.finish();
    }
    hs_caches_init(reinterpret_cast<hs_caches_t*>(caches.data()));
    reader_t reader{&words, 0};
    hs_decoder_t dec(reinterpret_cast<hs_caches_t*>(caches.data()), fetch, &reader);
    for (int i = 0; i < n; i++) {
      const auto x = dec.get(ips[i]);
      ASSERT_TRUE(x.lo == xs[i].lo && x.hi == xs[i].hi) << "trial " << trial << ", entry " << i;
    }
    // Again through take_run, in rows of varying length as the solver does
    hs_caches_init(reinterpret_cast<hs_caches_t*>(caches.data()));
    reader_t reader2{&words, 0};
    hs_decoder_t dec2(reinterpret_cast<hs_caches_t*>(caches.data()), fetch, &reader2);
    for (int i = 0; i < n;) {
      const int row = std::min(n - i, 1 + random.uniform<int>(0, 7));
      for (int c = 0; c < row;) {
        if (const int run = dec2.take_run(row - c)) {
          for (int j = 0; j < run; j++) {
            ASSERT_TRUE(xs[i+c+j].lo == ips[i+c+j].lo && xs[i+c+j].hi == ips[i+c+j].hi) << "trial " << trial << ", run entry " << i+c+j;
            dec2.push_history(ips[i+c+j]);
          }
          dec2.end_run(ips[i+c+run-1]);
          c += run;
        } else {
          const auto x = dec2.get(ips[i+c]);
          ASSERT_TRUE(x.lo == xs[i+c].lo && x.hi == xs[i+c].hi) << "trial " << trial << ", entry " << i+c;
          c++;
        }
      }
      i += row;
    }
  }
}

// The tiled solver must reproduce midsolve_internal exactly
TEST(tiled, matches_midsolve) {
  Random random(1234);
  const auto workspace = midsolve_workspace(18);
  for (const int trial : range(12)) {
    const int stones = random.uniform<int>(18, 24);
    const auto board = high_board_t::from_board(random_board(random, stones), trial & 1);
    const auto dense = midsolve_internal(board, workspace);
    tiled_options_t opts;
    opts.prefix = trial % 7;  // 0 means automatic
    opts.threads = trial % 3 == 0 ? 4 : 1;
    opts.merged = trial % 2 == 1;
    tiled_stats_t stats;
    const auto tiled = midsolve_tiled_internal(board, opts, &stats);
    for (const int i : range(dense.size()))
      for (const int j : range(2)) {
        const auto a = to_bits(j ? dense[i].notlose : dense[i].win), b = to_bits(j ? tiled[i].notlose : tiled[i].win);
        ASSERT_TRUE(a.lo == b.lo && a.hi == b.hi) << "trial " << trial << ", stones " << stones << ", prefix " << opts.prefix
                                                  << ", merged " << opts.merged << ", entry " << i << "," << j;
      }
    slog("stones %d prefix %d threads %d: peak %.1f MB (arena %.1f MB), entries %lld, %.2f s", stones, opts.prefix, opts.threads,
         stats.peak_bytes / double(1 << 20), stats.arena_bytes / double(1 << 20), (long long)stats.entries, stats.seconds);
  }
}

// midsolve_tiled should agree with midsolve at the values level too
TEST(tiled, values) {
  Random random(99);
  const auto workspace = midsolve_workspace(18);
  for (const int trial : range(4)) {
    const auto board = high_board_t::from_board(random_board(random, 18 + trial), trial & 1);
    const auto a = midsolve(board, workspace);
    tiled_options_t opts;
    const auto b = midsolve_tiled(board, opts);
    ASSERT_EQ(a.size(), b.size());
    for (const int i : range(a.size()))
      ASSERT_TRUE(a[i] == b[i]);
  }
}

}  // namespace
}  // namespace pentago
