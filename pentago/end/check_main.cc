// Verify the results of an MPI run against old out-of-core data

#include "pentago/end/config.h"
#include "pentago/base/all_boards.h"
#include "pentago/data/block_cache.h"
#include "pentago/data/file.h"
#include "pentago/data/numpy.h"
#include "pentago/data/supertensor.h"
#include "pentago/end/check.h"
#include "pentago/end/compacting_store.h"
#include "pentago/end/options.h"
#include "pentago/end/predict.h"
#include "pentago/end/sections.h"
#include "pentago/end/store_block_cache.h"
#include "pentago/end/simple_partition.h"
#include "pentago/end/verify.h"
#include "pentago/high/board.h"
#include "pentago/high/check.h"
#include "pentago/search/superengine.h"
#include "pentago/utility/log.h"
#include "pentago/utility/thread.h"
#include "pentago/utility/join.h"
#include "pentago/utility/mmap.h"
#include "pentago/utility/portable_hash.h"
#include <unordered_set>
#include <regex>
#include <vector>
#include <fnmatch.h>
#include <getopt.h>

namespace pentago {
namespace end {
namespace {

using std::make_shared;
using std::unordered_set;
using std::vector;

struct options_t {
  vector<string> dirs;
  string old = "../old-data-15july2012";
  bool restart = false;
  int reader_test = -1;
  int high_test = -1;
};

options_t parse_options(int argc, char** argv) {
  options_t o;
  static const option options[] = {
      {"help", no_argument, 0, 'h'},
      {"old", required_argument, 0, 'o'},
      {"restart", no_argument, 0, 's'},
      {"reader-test", required_argument, 0, 'r'},
      {"high-test", required_argument, 0, 'i'},
      {0, 0, 0, 0},
  };
  const int rank = 0;
  for (;;) {
    int option = 0;
    int c = getopt_long(argc, argv, "", options, &option);
    if (c == -1) break;  // Out of options
    switch (c) {
      case 'h':
        slog("usage: %s [options...] [dir]...", argv[0]);
        slog("Compare MPI results with out-of-core results");
        slog("  -h, --help                  Display usage information and quit");
        slog("      --old <dir>             Compare against an old directory (default '%s')", o.old);
        slog("      --restart               Check restart consistency");
        slog("      --reader-test <slice>   Check consistency of <slice> with smaller slices");
        slog("      --high-test <slice>     Check consistency of high level interface up to <slice>");
        exit(0);
      case 'o':
        o.old = optarg;
        break;
      case 's':
        o.restart = true;
        break;
      PENTAGO_INT_ARG('r', reader-test, reader_test)
      PENTAGO_INT_ARG('i', high-test, high_test)
      default:
        die("impossible option character %d", c);
    }
  }
  for (int i = optind; i < argc; i++)
    o.dirs.push_back(argv[i]);
  return o;
}

void toplevel(int argc, char** argv) {
  const auto o = parse_options(argc, argv);
  Scope scope("check mpi");
  init_threads(-1, -1);
  slog("dirs = %s", join(" ", o.dirs));
  for (const auto& dir : o.dirs) {
    Scope scope(tfm::format("dir %s", dir));

    // Check whether the data is meaningless
    int meaningless = 0;
    for (const auto& f : glob(dir + "/meaningless-*")) {
      GEODE_ASSERT(!meaningless);
      char* end;
      meaningless = strtol(rindex(f.data(), '-')+1, &end, 0);
      GEODE_ASSERT(!*end);
    }

    if (o.reader_test >= 0) {
      // Load given slice and treat as ground truth
      const auto cache_file = tfm::format("%s/slice-%d.pentago", dir, o.reader_test);
      slog("loading reader test data from %s", cache_file);
      const auto cache = reader_block_cache(open_supertensors(cache_file), 1<<30);
      set_block_cache(cache);
    } else if (meaningless) {
      // Generate meaningless data
      const auto sections = make_shared<sections_t>(meaningless, all_boards_sections(meaningless, 8));
      const auto partition = make_shared<simple_partition_t>(1, sections, false);
      Scope scope("meaningless");
      const auto store = make_shared<compacting_store_t>(estimate_block_heap_size(*partition, 0));
      const auto cache = store_block_cache(meaningless_block_store(partition, 0, 0, store), 1<<30);
      set_block_cache(cache);
    }

    // Memorize the hashes of .pentago files from meaningless unit tests
    static const tuple<string,Vector<string,2>> hashes[] = {
        {"empty.pentago", {"0e6efa078eacb8dd20787f09ec9b4693355bfda3", "2f500627ac2fe3f62d9670bfa3ce00cdd94537a8"}},
        {"s-s4-r0/slice-0.pentago", {"64ff1b2f49e25db61bf393c3de11ce74c7e52605", "5b677b0778edb5b6703e84fc85c002ed6f8c8085"}},
        {"s-s4-r17/slice-0.pentago", {"64ff1b2f49e25db61bf393c3de11ce74c7e52605", "7f2a056a7ca95129eaa1f2cc4fbd7cfefcab56bd"}},
        {"s-s4-r*/slice-1.pentago", {"00fd7a12b67bb05b891d15cda1682d5c072b93c1", "bbdf25459d2475f46683514f7ad1ae34224f2f64"}},
        {"s-s5-r0/slice-0.pentago", {"66b2d8a3327879fcb75b51644e4160a2066c605f", "ae37f9af95ab20fd644cc890b2ababf7e20964ca"}},
        {"s-s5-r17/slice-0.pentago", {"66b2d8a3327879fcb75b51644e4160a2066c605f", "f5ec8c9c3425443a981a0d4f406ad64883b2c8eb"}},
        {"s-s5-r*/slice-1.pentago", {"aa0d47a7c827b8ff8de50c16700e333f99602a39", "8c4ebf7867506c60ebd5d3f4a57ea7327d5c853d"}},
        {"s-s4-r0/slice-2.pentago", {"2a2e6bf617c8624c7fd6cb4a74ea42439102f329", "e9f829447980b2d427bf96002332704c65c37015"}},
        {"s-s4-r0/slice-3.pentago", {"72a769dcf4fa47f881d4997785a0134e50b610aa", "db33d83c472e53fcf669e8c75bd277f8c5caecf7"}},
        {"s-s4-r17/slice-2.pentago", {"f54b5c1bea1261f01b75da2eb0a2748d78eb6b39", "23bdbf069ecd68c01524a4b6075bcaba03613a9e"}},
        {"s-s4-r17/slice-3.pentago", {"7754a4235a373f5c5f00d78957e5caa996c27928", "9b06ad92f6a9ec982944d93438f8b40cb3cf763f"}},
        {"s-s5-r0/slice-2.pentago", {"ec29c1cc2d72eb92a2d204dcd92c8e3d37121d50", "6d1c0774631a5260b4ae2a22658f7f5b1036fc2e"}},
        {"s-s5-r0/slice-3.pentago", {"5c1bb0b5610cb8c46dec498b316aac7974ee055d", "f3ff4a5b2b78060c7d32d5c18b6f8ff9f6276f3d"}},
        {"s-s5-r0/slice-4.pentago", {"e0d48b631d3f12a61d33fea9822dc46ba70d8eb8", "3f6d11649725a13b11774f889e181642ac63b41d"}},
        {"s-s5-r17/slice-2.pentago", {"a9a6c589b4a58a5e773c62bc3d993a3000417c69", "6c0438587aaac9a7d64815f9d629f1efeaa86183"}},
        {"s-s5-r17/slice-3.pentago", {"3e1b9ab9bbd75ba9d16fd95b303b16a6fe413d24", "539650866f5e742555b74cc39553b4b181220245"}},
        {"s-s5-r17/slice-4.pentago", {"0a948c604ba9cc2d356ec125fc3d2131552eb478", "5bfb025e5729d39dc7785f75baae8099510656cc"}},
        {"write-3/slice-3.pentago", {"4241e562b57deecd6a948f682719f817e7cb44f2", "f12bc166507f4b31bf877f41583a989d32b3876c"}},
        {"write-4/slice-4.pentago", {"c42d21eaa6d881a7ef7c496e5ddbccc2f756d597", "bae84659c091f92add4f3fa7725210fb0a0212eb"}},
    };
    const auto check_hash = [&](const string& file) {
      if (!meaningless) return;  // We don't have nonmeaningless results memorized
      const std::regex restarted_re("-restarted");
      for (const auto& [p, hs] : hashes) {
        const auto file_sub = regex_replace(file, restarted_re, "");
        if (!fnmatch(("*" + p).c_str(), file_sub.c_str(), 0)) {
          const auto h2 = sha1(mmap_file(file));
          slog("sha1sum %s = %s", file, h2);
          const auto h = hs[pad_io];
          if (h != h2)
            throw ValueError(tfm::format("%s: expected sha1sum %s, got %s", file, h, h2));
          return;
        }
      }
      throw ValueError(tfm::format("don't know correct sha1sum for %s", file));
    };

    // Prepare to read data for high level test if necessary
    const auto high_data = [&]() {
      if (o.high_test < 0)
        return shared_ptr<const block_cache_t>();
      vector<shared_ptr<const supertensor_reader_t>> readers;
      for (const auto& f : glob(dir + "/slice-*.pentago"))
        extend(readers, open_supertensors(f));
      return reader_block_cache(readers, 1<<30);
    }();

    // Process each slice that exists, keeping track of which files we've checked
    const std::regex skip_pattern(R"(^(?:log(?:-\d+)?|\.{1,2}|empty.pentago|history.*|output-.*|.*\.pbs|meaningless-\d|\d+\.(cobaltlog|error|output))$)");
    unordered_set<string> unchecked;
    for (const auto& f : listdir(dir))
      if (!regex_match(f, skip_pattern))
        unchecked.insert(tfm::format("%s/%s", dir, f));
    Random random(8183131);
    init_supertable(16);
    for (int slice = 35; slice >= 0; slice--) {
      const auto empty_file = tfm::format("%s/empty.pentago", dir);
      const auto counts_file = tfm::format("%s/counts-%d.npy", dir, slice);
      const auto sparse_file = tfm::format("%s/sparse-%d.npy", dir, slice);
      const auto slice_file = tfm::format("%s/slice-%d.pentago", dir, slice);
      const auto restart_file = tfm::format("%s/slice-%d-restart.pentago", dir, slice);
      if (exists(counts_file)) {
        Scope scope(tfm::format("check slice %d", slice));

        // Read sparse samples
        const auto samples = read_numpy<uint64_t,9>(sparse_file);
        const Array<board_t> sample_boards(samples.size(), uninit);
        const Array<Vector<super_t,2>> sample_wins(samples.size(), uninit);
        for (const int i : range(samples.size())) {
          sample_boards[i] = samples[i][0];
          static_assert(sizeof(Vector<super_t,2>) == 8*sizeof(uint64_t));
          memcpy(&sample_wins[i], &samples[i][1], 8*sizeof(uint64_t));
        }
        unchecked.erase(sparse_file);

        // Read counts
        const auto sections_and_counts = read_numpy<uint64_t,4>(counts_file);
        GEODE_ASSERT(samples.size() == 256*sections_and_counts.size());
        const Array<section_t> sections(sections_and_counts.size(), uninit);
        const Array<Vector<uint64_t,3>> counts(sections.size(), uninit);
        for (const int i : range(sections.size())) {
          static_assert(sizeof(section_t) == sizeof(uint64_t));
          memcpy(&sections[i], &sections_and_counts[i][0], sizeof(uint64_t));
          memcpy(&counts[i], &sections_and_counts[i][1], 3*sizeof(uint64_t));
        }
        {
          string s = "sections =";
          for (const auto& section : sections)
            s += tfm::format(" %s", section);
          slog(s);
        }
        unchecked.erase(counts_file);

        // Compare against superengine
        {
          Scope scope("validity");
          endgame_sparse_verify(sample_boards, sample_wins, random, sample_boards.size());
        }

        // Run high level test if desired
        if (slice <= o.high_test) {
          Scope scope("high level test");
          sample_check(*high_data, sample_boards, sample_wins);
        }

        // Check empty.pentago hash
        check_hash(empty_file);

        // Open slice file
        if (exists(slice_file)) {
          check_hash(slice_file);
          const auto readers = open_supertensors(slice_file);
          uint64_t expected_size = 20 + 3*4;
          for (const auto& reader : readers)
            expected_size += reader->total_size();
          const auto actual_size = mmap_file(slice_file).size();
          if (pad_io)
            GEODE_ASSERT(expected_size <= size_t(actual_size));
          else
            GEODE_ASSERT(expected_size == size_t(actual_size));
          GEODE_ASSERT(size_t(counts.size()) == readers.size());
          unchecked.erase(slice_file);

          // Check each section and block
          if (o.reader_test < 0 || slice < o.reader_test) {
            Scope scope("consistency");
            for (const int i : range(readers.size())) {
              const auto reader = readers[i];
              const auto section = sections[i];
              const auto count = counts[i];
              GEODE_ASSERT(reader->header.section == section);
              const auto old_reader = meaningless || o.old.empty() ? nullptr :
                  make_shared<supertensor_reader_t>(tfm::format("%s/section-%s.pentago", o.old, section));
              const auto [count2, samples_found] = compare_readers_and_samples(
                  *reader, old_reader, sample_boards, sample_wins);
              GEODE_ASSERT(count == count2);
              GEODE_ASSERT(samples_found == 256);
            }
          }

          // Check restart consistency
          GEODE_ASSERT(o.restart == exists(restart_file));
          if (o.restart) {
            Scope scope("restart");
            const auto restarts = open_supertensors(restart_file);
            unchecked.erase(restart_file);
            GEODE_ASSERT(readers.size() == restarts.size());
            for (const int i : range(readers.size())) {
              const auto read0 = readers[i];
              const auto read1 = restarts[i];
              GEODE_ASSERT(read0->header.section == read1->header.section);
              const auto blocks = read0->header.blocks;
              for (const auto i0 : range(uint8_t(blocks[0])))
                for (const auto i1 : range(uint8_t(blocks[1])))
                  for (const auto i2 : range(uint8_t(blocks[2])))
                    for (const auto i3 : range(uint8_t(blocks[3]))) {
                      const auto b = vec(i0, i1, i2, i3);
                      GEODE_ASSERT(read0->read_block(b) == read1->read_block(b));
                    }
            }
          }
        } else {
          slog("WARNING: No slice file found, skipping consistency check for slice %d", slice);
        }
      }
    }

    // Yell if any files remain
    if (unchecked.size())
      die("strange files: %s", join(" ", unchecked));
  }
  slog("All tests passed!");
}

}  // namespace
}  // namespace end
}  // namespace pentago

int main(int argc, char** argv) {
  try {
    pentago::end::toplevel(argc, argv);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
