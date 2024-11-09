// MPI command line options
#pragma once

#include "pentago/base/section.h"
namespace pentago {
namespace end {

struct options_t {
  int threads = 0;
  int save = -100;
  int block_size = 8;
  int level = 26;
  int64_t memory_limit = 0;
  int gather_limit = 32;
  int line_limit = 32;
  int samples = 256;
  int ranks = -1;
  string dir;
  string restart;
  string test;
  int meaningless = 0;
  bool per_rank_times = false;
  int stop_after = 0;
  int randomize = 0;
  bool log_all = false;
  section_t section;
};

// Parse command line options
options_t parse_options(int argc, char** argv, const int ranks = -1, const int rank = 0);

// Signal an option parsing error
#define PENTAGO_OPTION_ERROR(...) \
  die_helper(rank ? "" : tfm::format("%s: %s", argv[0], tfm::format(__VA_ARGS__)))

// Macro for parsing integer arguments with getopt_long
#define PENTAGO_INT_ARG(short_opt, long_opt, var) \
  case short_opt: { \
    char* end; \
    const auto n = strtol(optarg, &end, 0); \
    if (!*optarg || *end) \
      PENTAGO_OPTION_ERROR("--" #long_opt " expected int, got '%s'", optarg); \
    if (__int128_t(n) != __int128_t(decltype(o.var)(n))) \
      PENTAGO_OPTION_ERROR("--" #long_opt " got %lld which doesn't fit in %s", n, typeid(n).name()); \
    o.var = decltype(o.var)(n); \
    break; }

}  // namespace end
}  // namespace pentago
