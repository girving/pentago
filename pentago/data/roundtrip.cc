// Read and write a supertensor file, in order to detect format drift

#include "pentago/data/supertensor.h"
#include "pentago/end/config.h"
#include "pentago/mpi/options.h"
#include "pentago/utility/log.h"
#include "pentago/utility/thread.h"
#include "pentago/utility/str.h"
#include <sys/stat.h>
#include <getopt.h>
#include <map>
#include <set>

namespace pentago {
namespace {

using std::make_shared;
using std::make_tuple;
using std::map;
using std::set;

struct options_t {
  string input, output;
};

options_t parse_options(int argc, char** argv) {
  options_t o;
  static const option options[] = {
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0},
  };
  const int rank = 0;
  for (;;) {
    int option = 0;
    int c = getopt_long(argc, argv, "", options, &option);
    if (c == -1) break;  // Out of options
    switch (c) {
      case 'h':
        slog("usage: %s [options...] <in.pentago> [out.pentago]", argv[0]);
        slog("Read and optionally write a supertensor file.");
        slog("  -h, --help                  Display usage information and quit");
        exit(0);
      default:
        die("impossible option character %d", c);
    }
  }
  const int nargs = argc - optind;
  if (nargs != 1 && nargs != 2)
    PENTAGO_OPTION_ERROR("expected 1-2 arguments <in.pentago> [out.pentago], got %d", nargs);
  o.input = argv[optind];
  o.output = nargs == 2 ? argv[optind + 1] : "";
  return o;
}

void toplevel(int argc, char** argv) {
  const auto o = parse_options(argc, argv);
  Scope scope("roundtrip");
  init_threads(-1, -1);

  // Unconditionally use the mpi standard parameters
  const int block_size = end::block_size;
  const int filter = 1;
  const int level = 26;

  const auto readers = open_supertensors(o.input);
  slog("supertensors in %s: %d", o.input, readers.size());
  for (const auto& r : readers) {
    slog("  version %d, section %s", r->header.version, r->header.section);
    GEODE_ASSERT(r->header.block_size == block_size);
    GEODE_ASSERT(r->header.filter == filter);
  }

  if (o.output.size()) {
    // Sort all reader blobs by offset
    const Vector<uint8_t,4> index(255, 255, 255, 255);
    struct blob_t { int r; Vector<uint8_t,4> block; uint64_t size; };
    map<uint64_t,blob_t> blobs;
    for (const int r : range(readers.size())) {
      blobs[readers[r]->header.index.offset] = blob_t({
          r, index, readers[r]->header.index.compressed_size});
      const auto& offset = readers[r]->offset;
      const auto& compressed_size = readers[r]->compressed_size_;
      for (const uint8_t i0 : range(offset.shape()[0]))
        for (const uint8_t i1 : range(offset.shape()[1]))
          for (const uint8_t i2 : range(offset.shape()[2]))
            for (const uint8_t i3 : range(offset.shape()[3]))
              blobs[offset(i0,i1,i2,i3)] = blob_t({r, vec(i0,i1,i2,i3), compressed_size(i0,i1,i2,i3)});
    }

    // Compute padding bytes
    vector<uint64_t> padding;
    {
      uint64_t next = multiple_supertensor_header_size(readers.size());
      for (const auto& p : blobs) {
        while (next < p.first)
          padding.push_back(next++);
        next += p.second.size;
      }
      struct stat st;
      GEODE_ASSERT(stat(o.input.c_str(), &st) == 0);
      while (next < uint64_t(st.st_size))
        padding.push_back(next++);
    }
    slog("padding = %d", padding.size());

    // Create writers
    Array<section_t> sections(readers.size());
    for (const int i : range(readers.size()))
      sections[i] = readers[i]->header.section;
    const auto writers = supertensor_writers(o.output, sections, block_size, filter, level,
                                             asarray(padding).copy());

    // Write blobs out in order
    for (const auto& p : blobs) {
      const int r = p.second.r;
      const auto block = p.second.block;
      if (block != index)
        writers[r]->write_block(block, readers[r]->read_block(block));
      else
        writers[r]->finalize();
    }
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
