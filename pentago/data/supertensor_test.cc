#include "pentago/base/section.h"
#include "pentago/base/superscore.h"
#include "pentago/data/supertensor.h"
#include "pentago/utility/hash.h"
#include "pentago/utility/mmap.h"
#include "pentago/utility/thread.h"
#include "pentago/utility/log.h"
#include "pentago/utility/temporary.h"
#include "gtest/gtest.h"
#include <unordered_map>
namespace pentago {
namespace {

using std::make_shared;
using std::unordered_map;

TEST(supertensor, supertensor) {
  init_threads(-1,-1);

  // Choose tiny parameters
  const section_t section({{2,0},{0,2},{1,1},{1,1}});
  const int block_size = 6;
  const int filter = 1;
  const int level = 6;

  // Prepare for writing
  tempdir_t tmp("supertensor");
  const string path = tmp.path + "/test.pentago";
  const auto writer = make_shared<supertensor_writer_t>(path, section, block_size, filter, level);
  const auto blocks = writer->header.blocks;
  ASSERT_EQ(blocks, vec<uint16_t>(2,2,3,3));

  // Generate random data
  uint128_t key = 187131;
  unordered_map<Vector<uint8_t,4>,Array<const Vector<super_t,2>,4>> data;
  for (const auto i : range(uint8_t(blocks[0]))) {
    for (const auto j : range(uint8_t(blocks[1]))) {
      for (const auto k : range(uint8_t(blocks[2]))) {
        for (const auto l : range(uint8_t(blocks[3]))) {
          const auto b = vec(i,j,k,l);
          const auto shape = writer->header.block_shape(b);
          const auto five = random_supers(key++, concat(shape, vec(2)));
          data[b] = Array<const Vector<super_t,2>,4>(
              shape, shared_ptr<const Vector<super_t,2>>(five.owner(),
                  reinterpret_cast<const Vector<super_t,2>*>(five.data())));
        }
      }
    }
  }

  // Write blocks out in arbitrary (hashed) order
  for (const auto [b, block] : data)
    writer->write_block(b, block.copy());
  writer->finalize();

  // Test exact hash to verify endian safety.  This relies on deterministic
  // compression, and therefore may fail in future.  This hash is for zlib-1.2.11.
  ASSERT_EQ(sha1(mmap_file(path)), "c0e71ffcb78564039f1e7ca62d385c0628b1687c");

  // Prepare for reading
  const auto reader0 = make_shared<const supertensor_reader_t>(path);
  const auto readers = open_supertensors(path);
  ASSERT_EQ(readers.size(), 1);
  const auto reader1 = readers[0];
  for (const auto& reader : {reader0, reader1}) {
    ASSERT_EQ(reader->header.section, section);
    ASSERT_EQ(reader->header.block_size, block_size);
    ASSERT_EQ(reader->header.filter, filter);
    ASSERT_EQ(reader->header.blocks, blocks);
  }

  // Read and verify data in a different order than we wrote it
  for (const auto& reader : {reader0, reader1}) {
    for (const auto i : range(uint8_t(blocks[0]))) {
      for (const auto j : range(uint8_t(blocks[1]))) {
        for (const auto k : range(uint8_t(blocks[2]))) {
          for (const auto l : range(uint8_t(blocks[3]))) {
            const auto b = vec(i,j,k,l);
            const auto block = reader->read_block(b);
            ASSERT_EQ(block, data[b]);
          }
        }
      }
    }
  }
  report_thread_times(total_thread_times().times);
}

}  // namespace
}  // namespace pentago
