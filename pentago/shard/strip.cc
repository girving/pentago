// Drop high slices from shard files
//
// Reads each input shard, keeps slices 0..max_slice (strip_shard in shard.h), checks the result, and
// writes it under the same file name in the output directory.  Inputs already at max_slice are only
// checked, which verifies shards stripped earlier (or by strip.py).  Every output is verified before it is
// written: it parses, its header matches the input except for max_slice, its groups are byte for byte
// the input's, and each group decodes to exactly the entry count the shard mapping predicts for this
// shard.  Used in 2026-11 to drop slice 18 from gs://pentago-shards once slice 18 was deleted.

#include "pentago/shard/shard.h"
#include "pentago/shard/parallel.h"
#include "pentago/data/file.h"
#include "pentago/utility/debug.h"
#include "pentago/utility/log.h"
#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>
#include <thread>
#include <vector>
namespace pentago {
namespace {

using std::vector;

static Array<const uint8_t> read_all(const string& path) {
  struct stat st;
  GEODE_ASSERT(stat(path.c_str(), &st) == 0, path);
  Array<uint8_t> buf(int(st.st_size), uninit);
  const auto err = read_local_file(path)->pread(buf, 0);
  GEODE_ASSERT(err.empty(), err);
  return buf;
}

static void toplevel(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: strip <max-slice> <output-dir> <shard-file> ...\n");
    exit(1);
  }
  const int max_slice = atoi(argv[1]);
  const string out_dir = argv[2];
  vector<string> paths(argv + 3, argv + argc);
  GEODE_ASSERT(0 <= max_slice && max_slice < 18);

  // Entry counts per slice depend only on the slice
  vector<uint64_t> totals;
  for (const int s : range(max_slice + 1))
    totals.push_back(shard_mapping_t(s).total());

  const int threads = int(std::thread::hardware_concurrency());
  parallel_for(threads, paths.size(), [&](const size_t i) {
    const auto& path = paths[i];
    const auto in = read_all(path);
    const shard_file_t sf(in);
    GEODE_ASSERT(int(sf.header.max_slice) >= max_slice, path);
    const bool done = int(sf.header.max_slice) == max_slice;
    const auto out = done ? in : Array<const uint8_t>(strip_shard(in, max_slice));

    // Verify before writing
    const shard_file_t so(out);
    const auto& h = so.header, &hi = sf.header;
    GEODE_ASSERT(int(h.max_slice) == max_slice && h.shard_id == hi.shard_id &&
                 h.total_shards == hi.total_shards && h.version == hi.version, path);
    GEODE_ASSERT(path.substr(path.rfind('/') + 1) == shard_filename(h.total_shards, h.shard_id), path);
    GEODE_ASSERT(so.group_offsets[max_slice + 1] == uint32_t(out.size()), path);
    const shard_locator_t loc(int(h.total_shards), range(0, 1));
    for (const int s : range(max_slice + 1)) {
      const auto a = in.slice(sf.group_offsets[s], sf.group_offsets[s + 1]);
      const auto b = out.slice(so.group_offsets[s], so.group_offsets[s + 1]);
      GEODE_ASSERT(a == b, tfm::format("%s: slice %d bytes differ", path, s));
      const auto n = arithmetic_decode(so.read_group(s)).size;
      GEODE_ASSERT(n == loc.shard_size(totals[s], int(h.shard_id)),
                   tfm::format("%s: slice %d has %d entries", path, s, n));
    }

    if (done) return;
    const auto out_path = tfm::format("%s/%s", out_dir, shard_filename(h.total_shards, h.shard_id));
    const auto err = write_local_file(out_path)->pwrite(out, 0);
    GEODE_ASSERT(err.empty(), err);
  });
  slog("checked %d shards at slices 0..%d", int(paths.size()), max_slice);
}

}  // namespace
}  // namespace pentago

int main(int argc, char** argv) {
  try {
    pentago::toplevel(argc, argv);
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
