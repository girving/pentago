// Compute total shard entries and estimate compressed shard sizes
//
// Uses section enumeration for exact entry counts, and counts-N.npy
// files for the ternary win/loss/tie distribution to estimate rANS
// entropy at various conditioning levels.

#include "pentago/base/all_boards.h"
#include "pentago/data/numpy.h"
#include "pentago/utility/log.h"
#include "pentago/utility/range.h"
#include <cmath>
#include <getopt.h>
namespace pentago {
namespace {

using std::max;

struct options_t {
  string dir = "data";
  int max_slice = 18;
};

static options_t parse_options(int argc, char** argv) {
  options_t o;
  static const option options[] = {
      {"help", no_argument, 0, 'h'},
      {"dir", required_argument, 0, 'd'},
      {"max-slice", required_argument, 0, 's'},
      {0, 0, 0, 0},
  };
  for (;;) {
    int option = 0;
    int c = getopt_long(argc, argv, "", options, &option);
    if (c == -1) break;
    switch (c) {
      case 'h':
        slog("usage: %s [options...]", argv[0]);
        slog("Compute shard entry counts and estimate compressed sizes.");
        slog("  -h, --help                Display usage information and quit");
        slog("  -d, --dir <path>          Directory containing counts-N.npy files (default: data)");
        slog("  -s, --max-slice <n>       Maximum slice (default: 18)");
        exit(0);
      case 'd':
        o.dir = optarg;
        break;
      case 's':
        o.max_slice = atoi(optarg);
        break;
      default:
        die("impossible option character %d", c);
    }
  }
  return o;
}

// Entropy contribution in bits for a single probability
static double entropy_bits(double p) {
  return p > 0 ? -p * log2(p) : 0;
}

// Entropy of a ternary distribution given (wins, losses, total)
static double ternary_entropy(uint64_t wins, uint64_t losses, uint64_t total) {
  if (total == 0) return 0;
  const double t = double(total);
  return entropy_bits(double(wins) / t) + entropy_bits(double(losses) / t)
       + entropy_bits(double(total - wins - losses) / t);
}

// Weighted entropy contribution in bits: total * H(distribution)
static double weighted_entropy(uint64_t wins, uint64_t losses, uint64_t total) {
  return double(total) * ternary_entropy(wins, losses, total);
}

void toplevel(int argc, char** argv) {
  const auto o = parse_options(argc, argv);
  GEODE_ASSERT(0 <= o.max_slice && o.max_slice <= 18);

  // Accumulate totals across all slices
  uint64_t cum_sections = 0;
  uint64_t cum_entries = 0;  // raw: section.size() * 256
  uint64_t cum_black_wins = 0;
  uint64_t cum_white_wins = 0;
  uint64_t cum_adjusted_total = 0;  // stabilizer-adjusted total from counts files
  double cum_slice_weighted_entropy = 0;     // Σ_slice entries_s * H_s
  double cum_section_weighted_entropy = 0;   // Σ_section entries_s * H_s

  slog("");
  slog("%-6s %8s %18s %18s %10s %10s %10s %8s", "slice", "sections", "slice_entries",
       "cum_entries", "p(bwin)", "p(wwin)", "p(tie)", "H(bits)");
  slog("%s", string(110, '-'));

  for (const int n : range(o.max_slice + 1)) {
    const auto sections = all_boards_sections(n, 8);
    uint64_t slice_entries = 0;
    for (const auto& s : sections)
      slice_entries += s.size() * 256;
    cum_sections += sections.size();
    cum_entries += slice_entries;

    // Read counts file for this slice
    const auto counts_file = tfm::format("%s/counts-%d.npy", o.dir, n);
    uint64_t slice_bwins = 0, slice_wwins = 0, slice_total = 0;
    const auto data = read_numpy<uint64_t, 4>(counts_file);
    GEODE_ASSERT(int(data.size()) == int(sections.size()),
                 tfm::format("counts-%d.npy has %d sections, expected %d",
                             n, data.size(), sections.size()));
    for (const int i : range(data.size())) {
      GEODE_ASSERT(data[i][0] == sections[i].sig(),
                   tfm::format("slice %d section %d signature mismatch: file %llu vs computed %llu",
                               n, i, data[i][0], sections[i].sig()));
      const uint64_t bw = data[i][1], ww = data[i][2], tot = data[i][3];
      slice_bwins += bw;
      slice_wwins += ww;
      slice_total += tot;

      // Per-section conditional entropy (using stabilizer-adjusted counts as proxy for raw)
      cum_section_weighted_entropy += weighted_entropy(bw, ww, tot);
    }

    cum_black_wins += slice_bwins;
    cum_white_wins += slice_wwins;
    cum_adjusted_total += slice_total;

    // Per-slice conditional entropy
    cum_slice_weighted_entropy += weighted_entropy(slice_bwins, slice_wwins, slice_total);

    const double pb = double(slice_bwins) / max(slice_total, uint64_t(1));
    const double pw = double(slice_wwins) / max(slice_total, uint64_t(1));
    const double pt = 1.0 - pb - pw;
    const double h = entropy_bits(pb) + entropy_bits(pw) + entropy_bits(pt);
    slog("  %3d  %8d %18llu %18llu %9.6f  %9.6f  %9.6f  %7.4f", n, int(sections.size()),
         slice_entries, cum_entries, pb, pw, pt, h);
  }

  slog("%s", string(110, '-'));
  slog("");

  // Summary
  const double entries_g = double(cum_entries) / 1e9;
  const double entries_t = double(cum_entries) / 1e12;
  slog("Total sections:     %d", int(cum_sections));
  slog("Total shard entries: %llu (%.3f G, %.6f T)", cum_entries, entries_g, entries_t);
  slog("Packed ternary:     %.3f GB (5 values/byte)", double(cum_entries) / 5.0 / 1e9);
  slog("");

  // Global entropy
  const double h_global = ternary_entropy(cum_black_wins, cum_white_wins, cum_adjusted_total);

  // Conditional entropies (weighted average of per-group entropies)
  const double h_slice = cum_slice_weighted_entropy / double(cum_adjusted_total);
  const double h_section = cum_section_weighted_entropy / double(cum_adjusted_total);

  slog("Global distribution (stabilizer-adjusted):");
  const double pb = double(cum_black_wins) / double(cum_adjusted_total);
  const double pw = double(cum_white_wins) / double(cum_adjusted_total);
  const double pt = 1.0 - pb - pw;
  slog("  black wins:  %llu (%.6f)", cum_black_wins, pb);
  slog("  white wins:  %llu (%.6f)", cum_white_wins, pw);
  slog("  ties:        %llu (%.6f)", cum_adjusted_total - cum_black_wins - cum_white_wins, pt);
  slog("  total:       %llu", cum_adjusted_total);
  slog("");

  slog("Entropy comparison:");
  slog("  H(value)           = %.6f bits/symbol  (single global distribution)", h_global);
  slog("  H(value | slice)   = %.6f bits/symbol  (per-slice distribution)", h_slice);
  slog("  H(value | section) = %.6f bits/symbol  (per-section distribution)", h_section);
  slog("  max = log2(3)      = %.6f bits/symbol", log2(3));
  slog("");

  // Size estimates: apply entropy rates to raw entry count
  const double bytes_global = double(cum_entries) * h_global / 8;
  const double bytes_slice = double(cum_entries) * h_slice / 8;
  const double bytes_section = double(cum_entries) * h_section / 8;

  slog("Estimated compressed sizes (%.3f T entries):", entries_t);
  slog("  Global distribution:       %8.3f TB", bytes_global / 1e12);
  slog("  Per-slice conditioning:    %8.3f TB  (saves %.1f%%)",
       bytes_slice / 1e12, 100 * (1 - bytes_slice / bytes_global));
  slog("  Per-section conditioning:  %8.3f TB  (saves %.1f%%)",
       bytes_section / 1e12, 100 * (1 - bytes_section / bytes_global));
  slog("");

  slog("For reference:");
  slog("  Current supertensor files (lzma): ~4.3 TB compressed");
  slog("  Uncompressed supertensors:        %.3f TB (64 bytes/position)",
       double(cum_entries) / 256 * 64 / 1e12);
  slog("  Naive ternary (log2(3) bits each): %.3f TB", double(cum_entries) * log2(3) / 8 / 1e12);

  // Shard count estimates
  slog("");
  slog("Shard sizing examples (for %llu total entries):", cum_entries);
  for (const int shards : {1000, 10000, 100000}) {
    const uint64_t per_shard = cum_entries / shards;
    slog("  %6d shards: %llu entries/shard", shards, per_shard);
    slog("           global rANS:  %.1f MB/shard", double(per_shard) * h_global / 8 / 1e6);
    slog("           per-section:  %.1f MB/shard", double(per_shard) * h_section / 8 / 1e6);
  }
}

}  // namespace
}  // namespace pentago

int main(int argc, char** argv) {
  try {
    pentago::toplevel(argc, argv);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
