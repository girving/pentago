#include "pentago/utility/aligned.h"
#include "pentago/utility/char_view.h"
#include "pentago/utility/debug.h"
#include "gtest/gtest.h"
#include <vector>
namespace pentago {

using std::vector;

// Most useful when run under valgrind to test for leaks
TEST(aligned_test, aligned) {
  // Check empty buffer
  struct large_t { char data[64]; };
  aligned_buffer<large_t>(0);

  vector<Array<uint8_t>> buffers;
  for (int i=0;i<100;i++) {
    // Test 1D
    auto x = aligned_buffer<int>(10);
    x.zero();
    x[0] = 1;
    x.back() = 2;
    ASSERT_EQ(x.sum(), 3);
    buffers.push_back(char_view_own(x));

    // Test 2D
    auto y = aligned_buffer<float>(vec(4,5));
    y.flat().zero();
    y(0,0) = 1;
    y(0,4) = 2;
    y(3,0) = 3;
    y(3,4) = 4;
    ASSERT_EQ(y.sum(), 10);
    buffers.push_back(char_view_own(y.flat_own()));
  }

  for (auto& x : buffers)
    ASSERT_EQ((((long)x.data())&15), 0);
}

}
