// Search for optimal L/H bit-level permutation shifts for each slice.
//
// 8-step LL-HH-LL-HH structure, no rotations. Each batch applies RSX then LSA.
// Uses chi-squared score on consecutive blocks of 256 as the optimization metric.
//
// Usage: bazel-bin/pentago/data/shard_permute_search

#include "pentago/data/shard_permute.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>

using std::max;
using std::mt19937_64;
using std::uniform_int_distribution;

namespace pentago {

static double chi2_score(const shard_permute_t& p, mt19937_64& rng, const int trials) {
  const int bins = 16, block = 256;
  const double expected = double(block) / bins;
  const uint64_t n = p.n;
  double chi2_sum = 0;
  uniform_int_distribution<uint64_t> dist(0, n - block - 1);
  for (int t = 0; t < trials; t++) {
    const uint64_t base = dist(rng);
    int counts[16] = {};
    for (int r = 0; r < block; r++)
      counts[int(double(p.forward(base + r)) / n * bins)]++;
    double chi2 = 0;
    for (int i = 0; i < bins; i++)
      chi2 += (counts[i] - expected) * (counts[i] - expected) / expected;
    chi2_sum += chi2;
  }
  return chi2_sum / trials;
}

static void search_slice(const int slice) {
  const uint64_t n = permute_constants[slice].n;
  const int k = 63 - __builtin_clzll(n);
  const uint64_t pow2k = uint64_t(1) << k;
  const double overlap = double(2 * pow2k - n) / n * 100;

  fprintf(stderr, "slice %d: n=%llu, k=%d, overlap=%.1f%%\n",
          slice, (unsigned long long)n, k, overlap);

  mt19937_64 rng(slice * 12345 + 67890);
  int best_shifts[PERM_STEPS];
  double best_score = -1e18;
  for (int i = 0; i < PERM_STEPS; i++)
    best_shifts[i] = max(1, (i + 1) * k / (PERM_STEPS + 1));

  // Larger k means slower forward calls, so use fewer trials but more restarts
  const int restarts = k > 30 ? 200 : 100;
  const int hill_steps = k > 30 ? 600 : 500;
  const int search_trials = k > 30 ? 100 : 200;

  for (int restart = 0; restart < restarts; restart++) {
    int shifts[PERM_STEPS];
    if (restart == 0) memcpy(shifts, best_shifts, sizeof(shifts));
    else for (int i = 0; i < PERM_STEPS; i++)
      shifts[i] = 1 + uniform_int_distribution<int>(0, k - 2)(rng);

    auto make_perm = [&]() {
      permute_constants_t c;
      c.n = n;
      for (int i = 0; i < PERM_STEPS; i++) c.shifts[i] = shifts[i];
      return shard_permute_t(c);
    };

    {
      const auto p = make_perm();
      const double chi2 = chi2_score(p, rng, search_trials);
      const double s = -fabs(chi2 - 15.0);
      if (s > best_score) {
        best_score = s; memcpy(best_shifts, shifts, sizeof(shifts));
        fprintf(stderr, "  restart %d init: chi2=%.1f\n", restart, chi2);
      }
    }

    for (int step = 0; step < hill_steps; step++) {
      const int idx = uniform_int_distribution<int>(0, PERM_STEPS - 1)(rng);
      const int old_shift = shifts[idx];
      shifts[idx] = 1 + uniform_int_distribution<int>(0, k - 2)(rng);
      if (shifts[idx] == old_shift) continue;
      const auto p = make_perm();
      const double chi2 = chi2_score(p, rng, search_trials);
      const double s = -fabs(chi2 - 15.0);
      if (s > best_score) {
        best_score = s; memcpy(best_shifts, shifts, sizeof(shifts));
        fprintf(stderr, "  restart %d step %d: chi2=%.1f\n", restart, step, chi2);
      } else {
        shifts[idx] = old_shift;
      }
    }
  }

  {
    permute_constants_t c;
    c.n = n;
    for (int i = 0; i < PERM_STEPS; i++) c.shifts[i] = best_shifts[i];
    const auto p = shard_permute_t(c);
    fprintf(stderr, "  FINAL: chi2=%.2f\n", chi2_score(p, rng, 2000));
  }

  printf("  {%llu, {", (unsigned long long)n);
  for (int i = 0; i < PERM_STEPS; i++) printf("%s%d", i ? "," : "", best_shifts[i]);
  printf("}},  // slice %d\n", slice);
}

}  // namespace pentago

int main() {
  printf("inline constexpr permute_constants_t permute_constants[19] = {\n");
  for (int s = 0; s < 19; s++)
    pentago::search_slice(s);
  printf("};\n");
  return 0;
}
