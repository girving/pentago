// Stream board_value_t entries from shard files to stdout
//
// Outputs raw uint64_t values (board_value_t.data) for consumption by Python via pipe.
// The OS pipe buffer (~64KB on Linux) provides natural backpressure: when the consumer
// stops reading, write() blocks and the process pauses automatically.

#include "pentago/shard/shard.h"
#include "pentago/utility/debug.h"
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <vector>
namespace pentago {
namespace {

using std::vector;

static void usage() {
  fprintf(stderr, "usage: iterate [options] <shard-file> ...\n");
  fprintf(stderr, "Stream board_value_t entries from shard files to stdout.\n");
  fprintf(stderr, "  -e, --epochs N   Number of epochs (default: 1, -1 = infinite)\n");
  fprintf(stderr, "  -s, --seed N     Random seed (default: 0)\n");
  fprintf(stderr, "  -h, --help       Display usage information and quit\n");
}

static void toplevel(int argc, char** argv) {
  int epochs = 1;
  uint128_t seed = 0;

  static const option options[] = {
      {"help", no_argument, 0, 'h'},
      {"epochs", required_argument, 0, 'e'},
      {"seed", required_argument, 0, 's'},
      {0, 0, 0, 0},
  };
  for (;;) {
    int option = 0;
    int c = getopt_long(argc, argv, "he:s:", options, &option);
    if (c == -1) break;
    switch (c) {
      case 'h': usage(); exit(0);
      case 'e': epochs = atoi(optarg); break;
      case 's': seed = uint128_t(strtoull(optarg, nullptr, 0)); break;
      default: die("impossible option character %d", c);
    }
  }

  // Collect shard filenames
  vector<string> paths;
  for (int i = optind; i < argc; i++)
    paths.emplace_back(argv[i]);
  if (paths.empty()) {
    fprintf(stderr, "error: no shard files specified\n\n");
    usage();
    exit(1);
  }

  // Iterate and write to stdout
  static constexpr int batch_size = 4096;
  shard_iterator_t it(std::move(paths), seed, epochs);
  const Array<board_value_t> batch(batch_size, uninit);
  while (!it.done()) {
    const int n = it.next_batch(batch);
    if (n == 0) break;
    const size_t written = fwrite(batch.data(), sizeof(board_value_t), n, stdout);
    if (int(written) != n)
      die("fwrite failed");
    fflush(stdout);
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
