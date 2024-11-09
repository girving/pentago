// Make .index.pentago files from .pentago files

#include "pentago/data/supertensor.h"
#include "pentago/end/sections.h"
#include "pentago/high/index.h"
#include "pentago/utility/thread.h"
namespace pentago {
namespace end {
namespace {

using std::make_shared;
using std::make_tuple;

Array<const uint8_t> pread(const string& path, const compact_blob_t& blob) {
  const auto f = read_local_file(path);
  Array<uint8_t> data(blob.size, uninit);
  const auto error = f->pread(data, blob.offset());
  if (!error.empty()) throw IOError(error);
  return data;
}

void toplevel() {
  Random random(183111);
  init_threads(-1, -1);
  for (const int slice : range(19)) {
    const auto name = tfm::format("../all/slice-%d.pentago", slice);
    const auto index_name = tfm::format("slice-%d.pentago.index", slice);
    const auto readers = open_supertensors(name);
    if (!exists(index_name))
      write_supertensor_index(index_name, readers);

    // Test
    const auto index = make_shared<supertensor_index_t>(descendent_sections(section_t(), slice)[slice]);
    for (__attribute__((unused)) int i : range(20)) {
      const auto r = readers[random.uniform<int>(readers.size())];
      const auto section = r->header.section;
      const auto offsets = r->block_offsets();
      const auto block = random.uniform(Vector<uint8_t,4>(), Vector<uint8_t,4>(offsets.shape()));
      // Read data the old way
      const Array<const Vector<super_t,2>,4> data = r->read_block(block);
      // Read using the index
      const auto blob = pread(index_name, index->blob_location(make_tuple(section, block)));
      const Array<const Vector<super_t,2>,4> data2 = index->unpack_block(
          make_tuple(section, block), pread(name, index->block_location(blob)));
      GEODE_ASSERT(data == data2);
    }
  }
}

}  // namespace
}  // namespace end
}  // namespace pentago

int main() {
  try {
    pentago::end::toplevel();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
