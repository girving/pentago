// MPI command line options

#include "pentago/mpi/options.h"
#include "pentago/end/config.h"
#include "pentago/utility/log.h"
#include <getopt.h>
namespace pentago {
namespace mpi {

using std::max;
using namespace pentago::end;
#define error PENTAGO_OPTION_ERROR

static const section_t bad_section(Vector<Vector<uint8_t,2>,4>(Vector<uint8_t,2>(255,0),Vector<uint8_t,2>(),Vector<uint8_t,2>(),Vector<uint8_t,2>()));

static section_t parse_section(const string& s) {
  section_t r;
  if (s.size() != 8)
    return bad_section;
  for (int i=0;i<4;i++)
    for (int j=0;j<2;j++) {
      const char c = s[2*i+j];
      if (c<'0' || c>'9')
        return bad_section;
      r.counts[i][j] = c-'0';
    }
  return r;
}

options_t parse_options(int argc, char** argv, const int ranks, const int rank) {
  options_t o;
  static const option options[] = {
      {"help", no_argument, 0, 'h'},
      {"threads", required_argument, 0, 't'},
      {"block-size", required_argument, 0, 'b'},
      {"save", required_argument, 0, 's'},
      {"dir", required_argument, 0, 'd'},
      {"restart", required_argument, 0, 'T'},
      {"level", required_argument, 0, 'l'},
      {"memory", required_argument, 0, 'm'},
      {"gather-limit", required_argument, 0, 'g'},
      {"line-limit", required_argument, 0, 'L'},
      {"samples", required_argument, 0, 'p'},
      {"ranks", required_argument, 0, 'r'},
      {"test", required_argument, 0, 'u'},
      {"meaningless", required_argument, 0, 'n'},
      {"per-rank-times", no_argument, 0, 'z'},
      {"stop-after", required_argument, 0, 'S'},
      {"randomize", required_argument, 0, 'R'},
      {"log-all", no_argument, 0, 'a'},
      {0, 0, 0, 0}
  };
  for (;;) {
    int option = 0;
    int c = getopt_long(argc, argv, "ht:b:s:d:m:", options, &option);
    if (c == -1) break;  // Out of options
    switch (c) {
      case 'h':
        if (!rank) {
          slog("usage: %s [options...] <section>", argv[0]);
          slog("options:");
          slog("  -h, --help                 Display usage information and quit");
          slog("  -t, --threads <threads>    Number of threads per MPI rank (required)");
          slog("  -b, --block-size <size>    4D block size for each section (default %d)", block_size);
          slog("  -s, --save <n>             Save all slices with n stones for fewer (required)");
          slog("  -d, --dir <dir>            Save and log to given new directory (required)");
          slog("      --restart <file>       Restart from the given slice file");
          slog("      --level <n>            Compression level: 1-9 is zlib, 20-29 is xz (default %d)", o.level);
          slog("  -m, --memory <n>           Approximate memory usage limit per *rank* (required)");
          slog("      --gather-limit <n>     Maximum number of simultaneous active line gathers (default %d)", o.gather_limit);
          slog("      --line-limit <n>       Maximum number of simultaneously allocated lines (default %d)", o.line_limit);
          slog("      --samples <n>          Number of sparse samples to save per section (default %d)", o.samples);
          slog("      --ranks <n>            Allowed for compatibility with predict, but must match mpirun --np");
          slog("      --test <name>          Run the MPI side of one of the unit tests");
          slog("      --meaningless <n>      Use meaningless values the given slice");
          slog("      --per-rank-times       Print a timing report for each rank");
          slog("      --stop-after <n>       Stop after computing the given slice");
          slog("      --randomize <key>      If nonzero, partition lines and blocks randomly using the given key");
          slog("      --log-all              Write log files for every process");
        }
        exit(0);
        break;
      PENTAGO_INT_ARG('t', threads, threads)
      PENTAGO_INT_ARG('b', block-size, block_size)
      PENTAGO_INT_ARG('s', save, save)
      PENTAGO_INT_ARG('l', level, level)
      PENTAGO_INT_ARG('p', samples, samples)
      PENTAGO_INT_ARG('r', ranks, ranks)
      PENTAGO_INT_ARG('n', meaningless, meaningless)
      PENTAGO_INT_ARG('g', gather-limit, gather_limit)
      PENTAGO_INT_ARG('L', line-limit, line_limit)
      PENTAGO_INT_ARG('S', stop-after, stop_after)
      PENTAGO_INT_ARG('R', randomize, randomize)
      case 'm': {
        char* end;
        double memory = strtod(optarg, &end);
        if (!strcmp(end, "MB") || !strcmp(end, "M"))
          o.memory_limit = uint64_t(memory*pow(2.,20));
        else if (!strcmp(end,"GB") || !strcmp(end,"G"))
          o.memory_limit = uint64_t(memory*pow(2.,30));
        else
          error("don't understand memory limit \"%s\", use e.g. 1.5GB",optarg);
        break; }
      case 'd':
        o.dir = optarg;
        break;
      case 'T':
        o.restart = optarg;
        break;
      case 'u':
        o.test = optarg;
        break;
      case 'z':
        o.per_rank_times = true;
        break;
      case 'a':
        o.log_all = true;
        break;
      default:
        error("impossible option character %d", c);
    }
  }
  if (argc-optind != 1 && !o.test.size())
    error("expected exactly one argument (section)");
  if (!o.threads)
    error("must specify --threads n (or -t n) with n > 1");
  if (o.threads < 1)
    error("need at least two threads for communication vs. compute");
  if (block_size != o.block_size)
    error("block size is currently hard coded to %d, can't specify %s", block_size, o.block_size);
  if (block_size < 2 || (block_size&1))
    error("invalid block size %d", block_size);
  if (block_size != 8)
    error("for now, block size is hard coded to 8 (see compute.cpp)");
  if (o.gather_limit < 1)
    error("--gather-limit %d must be at least 1", o.gather_limit);
  if (o.line_limit < 2)
    error("--line-limit %d must be at least 2", o.line_limit);
  if (o.samples < 0)
    error("must specify positive value for --samples, not %d", o.samples);
  if (o.save == -100 && !o.test.size())
    error("must specify how many slices to save with --save");
  if (ranks >= 0) {
    if (o.ranks >= 0 && o.ranks != ranks)
      error("--ranks %d doesn't match actual number of ranks %d", o.ranks, ranks);
    o.ranks = ranks;
  }
  if (o.stop_after < 0)
    error("--stop-after %d should be nonnegative", o.stop_after);
  if (!o.dir.size())
    error("must specify --dir", o.dir);
  if (o.meaningless > 9)
    error("--meaningless %d is outside valid range of [1,9]", o.meaningless);
  const char* section_str = argv[optind++];
  o.section = o.test.size() ? section_t() : parse_section(section_str);
  if (!o.section.valid())
    error("invalid section '%s'", section_str);
  if (o.section.sum() == 36)
    error("refusing to compute a 36 stone section");
  return o;
}

}
}
