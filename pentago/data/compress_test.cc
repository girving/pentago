#include "pentago/data/compress.h"
#include "pentago/utility/random.h"
#include "pentago/utility/thread.h"
#include "gtest/gtest.h"
namespace pentago {
namespace {

void compress_test(const int level) {
  init_threads(-1, -1);
  Random random(level);
  Array<uint8_t> input(random.uniform(10000), uninit);
  for (auto& c : input)
    c = random.bits<uint8_t>();
  const auto small = compress(input, level, unevent);
  const auto output = decompress(small, input.size(), unevent);
  ASSERT_EQ(input, output);
}

TEST(compress, zlib) { compress_test(6); }
TEST(compress, lzma) { compress_test(26); }

}  // namespace
}  // namespace pentago
